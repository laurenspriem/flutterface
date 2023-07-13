import 'dart:math' show Random, sqrt;

// run `dart test test/embedding_brute_search_test.dart ` to test
void main() {
  const sampleSize = 200000;
  testEmbeddingBruteSearch(sampleSize, 512);
}

void testEmbeddingBruteSearch(int sampleSize, int embeddingLength) {
  final embeddings = List.generate(
    sampleSize,
    (_) =>
        List.generate(embeddingLength, (_) => -1 + 2 * Random().nextDouble()),
  );
  final x =
      List.generate(embeddingLength, (_) => -1 + 2 * Random().nextDouble());

  final stopwatchEmbeddingBruteSearch = Stopwatch()..start();

  var minIndex = 0;
  var minDist = double.infinity;
  var minEmbedding = List.filled(embeddingLength, 0.0);
  for (var i = 0; i < embeddings.length; i++) {
    final dist = cosineDistance(embeddings[i], x);
    if (dist < minDist) {
      minDist = dist;
      minIndex = i;
      minEmbedding = embeddings[i];
    }
  }

  stopwatchEmbeddingBruteSearch.stop();
  print(
    'Embedding brute search for $sampleSize embeddings ($embeddingLength size) executed in ${stopwatchEmbeddingBruteSearch.elapsedMilliseconds}ms',
  );

  print('Embedding with the lowest distance is at index: $minIndex');
  print('Lowest distance: $minDist');
  print('Lowest embedding (first five numbers): ${minEmbedding.sublist(0, 5)}');
}

double cosineDistance(List<double> a, List<double> b) {
  assert(a.length == b.length);
  var sumXX = 0.0, sumYY = 0.0, sumXY = 0.0;

  for (var i = 0; i < a.length; i++) {
    sumXX += a[i] * a[i];
    sumYY += b[i] * b[i];
    sumXY += a[i] * b[i];
  }

  return 1 - sumXY / (sqrt(sumXX * sumYY));
}
