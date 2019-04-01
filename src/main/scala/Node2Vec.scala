import java.util.Random

import org.apache.log4j.Logger
import org.apache.log4j.Level
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.random.UniformGenerator
import breeze.linalg.{DenseMatrix, DenseVector, Vector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}


object Node2Vec {

  def main(args: Array[String]): Unit = {

    Logger.getLogger("org.apache.spark").setLevel(Level.ERROR)

    val data_path = args(0)

    val spark = SparkSession
      .builder()
      .appName("Node2Vec")
      .getOrCreate()

    val sc = spark.sparkContext

    def read_data(path: String, spark: SparkSession): RDD[(Int, Int)] = {
      spark.read.format("csv")
        .option("header", "true")
        .schema(StructType(Array(
          StructField("source_node", IntegerType, false),
          StructField("destination_node", IntegerType, false)
        )))
        .load(path).rdd.map(row => (row.getAs[Int](0), row.getAs[Int](1)))
    }

    val data = read_data(data_path, spark)

    def create_embedding_matrix(vocab_size: Int, emb_size: Int) = {
      val gen = new UniformGenerator()
      val data = Array.fill(vocab_size*emb_size)(gen.nextValue().toFloat - 0.5f / emb_size)
      new DenseMatrix(emb_size, vocab_size, data)
    }

    def blank_gradient_matrix(vocab_size: Int, emb_size: Int) = {
      val data = Array.fill(vocab_size*emb_size)(0.0f)
      new DenseMatrix(emb_size, vocab_size, data)
    }

    val batch_size = 1024
    val n_data = data.count()
    val n_partitions: Int = (n_data / batch_size).toInt
    val total_nodes = 1000000 //943150
    val emb_dim = 100

    val in_emb = create_embedding_matrix(total_nodes, emb_dim)
    val out_emb = create_embedding_matrix(total_nodes, emb_dim)



    def add_batch_index(data: RDD[(Int, Int)], batch_size: Int) = {
      // add index, and then swap so that index can be used as key
      data.zipWithIndex().map(x => ((x._2 / batch_size).toInt, x._1))
    }

    val data_with_batch_index = add_batch_index(data, batch_size)

    def partition(data: RDD[(Int, (Int, Int))], partition_index: Int) = {
      data.filter(_._1 == partition_index).map(_._2)
    }

    val batches = for (i <- 0 to n_partitions-1) yield partition(data_with_batch_index, i)

    def gradients(source: Int,
                  destination: Int,
                  emb_in: DenseMatrix[Float],
                  emb_out: DenseMatrix[Float],
                  total_nodes: Int,
                  neg_s: Int
                 ) = {
      val s_voc = emb_in.rows
      val s_emb = emb_in.cols

      def sigm(x:Double) = {
        (1 / (1 + Math.exp(-x)))
      }

      def pair_gradients(source: Int,
                         destination:Int,
                         label: Float) = {
        val in = emb_in(::, source)
        val out = emb_out(::, destination)

        val act = sigm(in.t * out).toFloat
        val grad_in = label * act * (1 - act) * out
        val grad_out = label * act * (1 - act) * in
        ((source, grad_in), (destination, grad_out))
      }

      val (pos_in_grads, pos_out_grads) = pair_gradients(source, destination, 1.0f)
      var in_grads = Array(pos_in_grads)
      var out_grads = Array(pos_in_grads)

      val rnd = new Random()

      for (i <- 1 to neg_s) {
        val (n_in_grads, n_out_grads) = pair_gradients(source, rnd.nextInt(total_nodes), -1.0f)
        in_grads :+= n_in_grads
        out_grads :+= n_out_grads
      }

      (in_grads, out_grads)
    }

    val learning_rate = 0.02f

    batches.foreach{ batch => {
      val in_broad = sc.broadcast(in_emb)
      val out_broad = sc.broadcast(out_emb)

      val grads = batch.map( pair => gradients(pair._1, pair._2, in_broad.value, out_broad.value, total_nodes, 20))
      val in_grads: RDD[(Int, DenseVector[Float])] = grads.flatMap(x => x._1)
      val out_grads: RDD[(Int, DenseVector[Float])] = grads.flatMap(x => x._2)

      def average_gradients(grads: RDD[(Int, DenseVector[Float])]): RDD[(Int, DenseVector[Float])] = {
        grads.map(x => (x._1, (x._2 , 1:Int)))
          .reduceByKey((r1, r2)=> (r1._1 + r2._1, r1._2 + r2._2))
          .map( x => (x._1, x._2._1 / x._2._2.toFloat))
      }

      val a_in_grad = average_gradients(in_grads).collectAsMap()
      val a_out_grad = average_gradients(out_grads).collectAsMap()

      val in_blank = blank_gradient_matrix(total_nodes, emb_dim)
      val out_blank = blank_gradient_matrix(total_nodes, emb_dim)

      for (k <- a_in_grad.keys){
        in_blank(::, k) := a_in_grad(k)
      }

      for (k <- a_out_grad.keys){
        out_blank(::, k) := a_out_grad(k)
      }

      in_emb += in_blank * learning_rate
      out_emb += out_blank * learning_rate

    }
    }


//    val repartitioned_for_batches = data.repartition(n_partitions)
//    data.zipWithIndex().map(x => ((x._2 / batch_size).toInt, x._1)).groupByKey().lookup(0).foreach(println)



  }
}