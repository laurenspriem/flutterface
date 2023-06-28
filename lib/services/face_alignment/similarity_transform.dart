import 'package:flutterface/extensions/ml_linalg_extensions.dart';
import 'package:ml_linalg/linalg.dart';

class SimilarityTransform {
  late Matrix params;

  bool estimate(List<List<double>> src, List<List<double>> dst) {
    params = _umeyama(src, dst, true);
    // We check for NaN in the transformation matrix params.
    final isNoNanInParam =
        !params.asFlattenedList.any((element) => element.isNaN);
    return isNoNanInParam;
  }

  static Matrix _umeyama(
    List<List<double>> src,
    List<List<double>> dst,
    bool estimateScale,
  ) {
    final srcMat = Matrix.fromList(src);
    final dstMat = Matrix.fromList(dst);
    final num = srcMat.rowCount;
    final dim = srcMat.columnCount;

    // Compute mean of src and dst.
    final srcMean = srcMat.mean(Axis.columns);
    final dstMean = dstMat.mean(Axis.columns);

    // Subtract mean from src and dst.
    final srcDemean = srcMat.mapRows((vector) => vector - srcMean);
    final dstDemean = dstMat.mapRows((vector) => vector - dstMean);

    // Eq. (38).
    final A = (dstDemean.transpose() * srcDemean) / num;

    // Eq. (39).
    var d = Vector.filled(dim, 1.0);
    if (A.determinant() < 0) {
      d = d.set(dim - 1, -1);
    }

    var T = Matrix.identity(dim + 1);

    final svdResult = A.svd();
    final Matrix U = svdResult['U']!;
    final Vector S = svdResult['S']!;
    final Matrix V = svdResult['V']!;

    // Eq. (40) and (43).
    final rank = A.matrixRank();
    if (rank == 0) {
      return T * double.nan;
    } else if (rank == dim - 1) {
      if (U.determinant() * V.determinant() > 0) {
        T = T.setSubMatrix(0, dim, 0, dim, U * V);
      } else {
        final s = d[dim - 1];
        d = d.set(dim - 1, -1);
        final replacement = U * Matrix.diagonal(d.toList()) * V;
        T = T.setSubMatrix(0, dim, 0, dim, replacement);
        d = d.set(dim - 1, s);
      }
    } else {
      final replacement = U * Matrix.diagonal(d.toList()) * V;
      T = T.setSubMatrix(0, dim, 0, dim, replacement);
    }

    var scale = 1.0;
    if (estimateScale) {
      // Eq. (41) and (42).
      scale = 1.0 / srcDemean.variance(Axis.columns).sum() * (S * d).sum();
    }

    final subTIndices = Iterable<int>.generate(dim, (index) => index);
    final subT = T.sample(rowIndices: subTIndices, columnIndices: subTIndices);
    final newSubT = dstMean - (subT * srcMean) * scale;
    T = T.setValues(0, dim, dim, dim + 1, newSubT);
    final newNewSubT =
        T.sample(rowIndices: subTIndices, columnIndices: subTIndices) * scale;
    T = T.setSubMatrix(0, dim, 0, dim, newNewSubT);

    return T;
  }
}
