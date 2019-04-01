import java.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.random.UniformGenerator
import breeze.linalg.{DenseMatrix, DenseVector, Vector, sum}
import breeze.numerics.log
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object utils {
  def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
    spark.read.format("csv")
      .option("header", "true")
      .schema(StructType(Array(
        StructField("source_node", IntegerType, false),
        StructField("destination_node", IntegerType, false)
      )))
      .load(path).rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
  }

  def create_embedding_matrix(vocab_size: Int, emb_size: Int) = {
    val gen = new UniformGenerator()
    val data = Array.fill(vocab_size*emb_size)(gen.nextValue().toFloat - 0.5f)// / emb_size)
    new DenseMatrix(emb_size, vocab_size, data)
  }

  def blank_gradient_matrix(vocab_size: Int, emb_size: Int) = {
    val data = Array.fill(vocab_size*emb_size)(0.0f)
    new DenseMatrix(emb_size, vocab_size, data)
  }

  def create_random_vector(vector_size: Int) = {
    val gen = new UniformGenerator()
    val data = Array.fill(vector_size)(gen.nextValue().toFloat - 0.5f)// / emb_size)
    new DenseVector(data)
  }

  def create_distributed_matrix(n_rows: Int, n_columns: Int, spark: SparkContext) = {
    spark.parallelize(0 to n_rows-1).map((_, create_random_vector(n_columns)))
  }

  def add_batch_index(data: RDD[(Int, Int)], batch_size: Int) = {
    // add index, and then swap so that index can be used as key
    data.zipWithIndex().map(x => ((x._2 / batch_size).toInt, x._1))
  }

  def partition(data: RDD[(Int, Int)], batch_size: Int) = {
    val data_with_batch_index = utils.add_batch_index(data, batch_size)
    val n_data = data.count()
    val n_partitions: Int = (n_data / batch_size).toInt + 1
    for (i <- 0 to n_partitions-1)
      yield data_with_batch_index.filter(_._1 == i).map(_._2)
  }
}


object nn {
  def sigm(x: Double ) = {
    (1 / (1 + Math.exp(-x)))
  }

  def loss(x: Double) = {
    -log(sigm(x))
//    -log(1 + Math.exp(-x))
  }

  def gradients(source: Int,
                destination: Int,
                emb_in: DenseMatrix[Float],
                emb_out: DenseMatrix[Float],
                neg_s: Int
               ) = {
    val s_voc = emb_in.rows
    val s_emb = emb_in.cols

    def pair_gradients(source: Int,
                       destination:Int,
                       label: Float) = {
      val in = emb_in(::, source)
      val out = emb_out(::, destination)

      val act = nn.sigm(in.t * out * label).toFloat
      val grad_in =  - label * out * (1 - act)
      val grad_out = - label * in * (1 - act)
      ((source, grad_in), (destination, grad_out))
    }

    val (pos_in_grads, pos_out_grads) = pair_gradients(source, destination, 1.0f)
    var in_grads = Array(pos_in_grads)
    var out_grads = Array(pos_out_grads)

    val rnd = new Random()
    val total_nodes = emb_in.rows

    for (i <- 1 to neg_s) {
      val (n_in_grads, n_out_grads) = pair_gradients(source, rnd.nextInt(total_nodes), -1.0f)
      in_grads :+= n_in_grads
      out_grads :+= n_out_grads
    }

    (in_grads, out_grads)
  }

  def collect_gradients(grads: RDD[(Int, DenseVector[Float])]) = {
    val n_items =  grads.count().toFloat
    grads.reduceByKey(_+_)
      .map(x => (x._1, x._2 / n_items))
      .collectAsMap()
  }

  def estimate_likelihood(source: Int,
                          destination: Int,
                          emb_in: DenseMatrix[Float],
                          emb_out:DenseMatrix[Float]) = {
    val in = emb_in(::, source)
    val out = emb_out(::, destination)

    nn.loss(in.t * out)
  }
}



object Node2Vec {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val data_path = args(0)
    val test_data = args(1)

    val spark = SparkSession
      .builder()
      .appName("Node2Vec")
//      .master("local")
      .getOrCreate()

    val sc: SparkContext = spark.sparkContext

    val data = utils.read_data(data_path, spark)
    val test = utils.read_data(test_data, spark)
//    val data = sc.parallelize(utils.read_data(data_path, spark).take(10))



    val batch_size = 50000
    val total_nodes = 1000000 //943150
    val emb_dim = 100
    val learning_rate = 10.0f

    val in_emb = utils.create_embedding_matrix(total_nodes, emb_dim)
    val out_emb = utils.create_embedding_matrix(total_nodes, emb_dim)
//    val in_emb = utils.create_distributed_matrix(total_nodes, emb_dim, sc)
//    val out_emb = utils.create_distributed_matrix(total_nodes, emb_dim, sc)

    val batches =  utils.partition(data, batch_size)



    for (batch <- batches) {
      val in_broad = sc.broadcast(in_emb)
      val out_broad = sc.broadcast(out_emb)

      val likelihood = test
        .map(x => nn.estimate_likelihood(x._1, x._2, in_broad.value, out_broad.value))
        .reduce(_+_)

      println(s"Loss ${likelihood / test.count()}")

//      println(s"${batch.count()}")

//      println(s"Check sum ${sum(in_emb)}")

      val grads = batch.map( pair => nn.gradients(pair._1, pair._2, in_broad.value, out_broad.value, 20))

      val in_grads = grads.flatMap(x => x._1)
      val out_grads = grads.flatMap(x => x._2)

      val a_in_grad = nn.collect_gradients(in_grads)
      val a_out_grad = nn.collect_gradients(out_grads)

      val in_blank = utils.blank_gradient_matrix(total_nodes, emb_dim)
      val out_blank = utils.blank_gradient_matrix(total_nodes, emb_dim)

      for (k <- a_in_grad.keys){
        in_blank(::, k) := a_in_grad(k)
      }

      for (k <- a_out_grad.keys){
        out_blank(::, k) := a_out_grad(k)
      }

//      println(s"In Check sum ${sum(in_blank)}")
//      println(s"Out Check sum ${sum(out_blank)}")

      in_emb -= in_blank * learning_rate
      out_emb -= out_blank * learning_rate



    }



//    val repartitioned_for_batches = data.repartition(n_partitions)
//    data.zipWithIndex().map(x => ((x._2 / batch_size).toInt, x._1)).groupByKey().lookup(0).foreach(println)



  }
}