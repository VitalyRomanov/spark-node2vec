import java.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.random.UniformGenerator
import breeze.linalg.{DenseMatrix, DenseVector}
import breeze.numerics.log
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

object utils {
  def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
    spark.read.format("csv")
      // the original data is store in CSV format
      // header: source_node, destination_node
      // here we read the data from CSV and export it as RDD[(Int, Int)],
      // i.e. as RDD of edges
      .option("header", "true")
      // State that the header is present in the file
      .schema(StructType(Array(
        StructField("source_node", IntegerType, false),
        StructField("destination_node", IntegerType, false)
      )))
      // Define schema of the input data
      .load(path)
      // Read the file as DataFrame
      .rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
      // Interpret DF as RDD
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

//  def sigm(x: Float ) = {
//    (1 / (1 + Math.exp(-x)))
//  }

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

    println("Usage: train test emb_dim learning_rate epochs")

    val batch_size = 10000
    val total_nodes = 40334 //943150
    val emb_dim = args(2).toInt
    val learning_rate = args(3).toFloat
    val epochs = args(4).toInt


    println("Reading data")

    var data = utils.read_data(data_path, spark)
    val test = utils.read_data(test_data, spark).repartition(100)
//    val data = sc.parallelize(utils.read_data(data_path, spark).take(10))


    val in_emb = utils.create_embedding_matrix(total_nodes, emb_dim)
    val out_emb = utils.create_embedding_matrix(total_nodes, emb_dim)

    println("Creating batches")

    val batches =  utils.partition(data, batch_size)
//    val batches =  utils.partition(sc.parallelize(data.take(100)), batch_size)

    data = null

    println("Begin training")

    for (e <- 1 to epochs) {
      for (batch <- batches) {
        var in_broad = sc.broadcast(in_emb)
        var out_broad = sc.broadcast(out_emb)

        var likelihood = test
          .map(x => nn.estimate_likelihood(x._1, x._2, in_broad.value, out_broad.value))
          .reduce(_ + _)

        println(s"Epoch $e Loss ${likelihood / test.count()}")

        //      println(s"${batch.count()}")

        //      println(s"Check sum ${sum(in_emb)}")

        var grads = batch.repartition(100)
          .map(pair => nn.gradients(pair._1, pair._2, in_broad.value, out_broad.value, 20))

        def gradient_updates(grads: RDD[(Int, DenseVector[Float])]) = {
          val a_grad = nn.collect_gradients(grads)
          val blank = utils.blank_gradient_matrix(total_nodes, emb_dim)

          for (k <- a_grad.keys) {
            blank(::, k) := a_grad(k)
          }

          blank
        }

        in_emb -= gradient_updates(grads.flatMap(x => x._1)) * learning_rate
        out_emb -= gradient_updates(grads.flatMap(x => x._2)) * learning_rate


      }
    }

    def get_top(source: Int, top_k: Int) = {
      var act = Array[(Int,Float)]()

      for (destination <- 0 to total_nodes-1) {
//      for (destination <- 0 to 100) {
        if (source != destination)
          act :+= (destination, in_emb(::, source).t * out_emb(::, destination))
      }

      val r = act.sortBy(_._2).takeRight(top_k).reverse

      (r.map(_._1), r.map(_._2))
    }

    val results = sc.parallelize(0 to total_nodes-1).repartition(100)
      .map( x => (x, get_top(x, 10)))

    results.map(x => (x._1, x._2._1.mkString(" ")))
      .map(x => s"${x._1}, ${x._2}")
      .saveAsTextFile("result_destination.csv")

    results.map(x => (x._1, x._2._2.mkString(" ")))
      .map(x => s"${x._1}, ${x._2}")
      .saveAsTextFile("result_score.csv")

  }
}