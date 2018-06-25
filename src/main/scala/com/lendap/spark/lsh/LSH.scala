package com.lendap.spark.lsh

/**
 * Created by maruf on 09/08/15.
 */

import java.io.FileWriter

import org.apache.spark.api.java.StorageLevels
import org.apache.spark.mllib.linalg.SparseVector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import java.io.File

/** Build LSH model with data RDD. Hash each vector number of hashTable times and stores in a bucket.
  *
  * @param data          RDD of sparse vectors with vector Ids. RDD(vec_id, SparseVector)
  * @param m             max number of possible elements in a vector
  * @param numHashFunc   number of hash functions
  * @param numHashTables number of hashTables.
  *
  * */
class LSH(data: RDD[(Long, SparseVector)] = null, m: Int = 0, numHashFunc: Int = 4, numHashTables: Int = 4, sc: SparkSession, modelOut : String) extends Serializable {


  def run(): RDD[((Long, String), Int)] = {
    //create a new model object

    val dataRDD = data.persist(StorageLevels.DISK_ONLY)



    var datasets = (0 until numHashFunc * numHashTables).map(i => {
      val hasher = Hasher(m)
      saveHasher(modelOut+i, hasher)

      dataRDD
        .map(v => ((v._1, i % numHashTables), hasher.hash(v._2)))

    })
    val rdd = sc.sparkContext.emptyRDD[((Long, Int), Int)]
    datasets.foldLeft(rdd)((dataset1, dataset2) => dataset1.union(dataset2))
      .groupByKey()
      .map(x => ((x._1._1, x._2.mkString("")), x._1._2))
  }

  def cosine(a: SparseVector, b: SparseVector): Double = {
    val intersection = a.indices.intersect(b.indices)
    val magnitudeA = intersection.map(x => Math.pow(a.apply(x), 2)).sum
    val magnitudeB = intersection.map(x => Math.pow(b.apply(x), 2)).sum
    intersection.map(x => a.apply(x) * b.apply(x)).sum / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB))
  }

  def saveHasher(filename: String, hasher: Hasher) = {
    val writer = new FileWriter(new File(filename))
    hasher.r.foreach(value => writer.write(value.toString + " "))
  }

}
