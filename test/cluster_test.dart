import 'dart:math' show Random;

import 'package:simple_cluster/src/dbscan.dart';
import 'package:test/test.dart';

// run `dart test test/cluster_test.dart ` to test
void main() {
  // Test simple_cluster package with DBSCAN
  final clusterOutput = testSimpleCluster(100, 192, 4, 20);

  group('Test simple_cluster package with DBSCAN', () {
    test('Test correct number of clusters', () {
      expect(clusterOutput.length, 4);
    });
  });
}

List<List<int>> testSimpleCluster(
  int sampleSize,
  int embeddingLength,
  int minimumNumberOfClusters,
  int minimumClusterSize,
) {
  final embeddings = List.generate(
    sampleSize,
    (_) =>
        List.generate(embeddingLength, (_) => -1 + 2 * Random().nextDouble()),
  );

  for (var i = 0; i < minimumNumberOfClusters; i++) {
    final x =
        List.generate(embeddingLength, (_) => -1 + 2 * Random().nextDouble());
    for (var j = 0; j < minimumClusterSize; j++) {
      embeddings.add(x);
    }
  }

  final stopwatchClustering = Stopwatch()..start();

  final DBSCAN dbscan = DBSCAN(
    epsilon: 3,
    minPoints: 2,
  );

  final List<List<int>> clusterOutput = dbscan.run(embeddings);

  stopwatchClustering.stop();
  print(
    'Clustering for $sampleSize embeddings ($embeddingLength size) executed in ${stopwatchClustering.elapsedMilliseconds}ms',
  );

  print("===== 1 =====");
  print("Clusters output");
  print(clusterOutput); //or dbscan.cluster
  print("Noise");
  print(dbscan.noise);
  print("Cluster label for points");
  print(dbscan.label);

  return clusterOutput;
}
