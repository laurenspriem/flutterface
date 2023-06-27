import 'dart:math' as math;

import 'package:matrix2d/matrix2d.dart';

typedef Matrix = List<List<double>>;

class SimilarityTransform {
  late Matrix params;
  static const m2d = Matrix2d();

  bool estimate(List<List<double>> src, List<List<double>> dst) {
    params = _umeyama(src, dst, true);
    // We check for NaN in the transformation matrix params.
    final isNoNanInParam =
        !params.flatten.any((element) => element.isNaN);
    return isNoNanInParam;
  }

  static Matrix _umeyama(
    List<List<double>> src,
    List<List<double>> dst,
    bool estimateScale,
  ) {
    final srcMat = src; // Matrix.fromList(src);
    final dstMat = dst; // Matrix.fromList(dst);
    final num = srcMat.length;
    final dim = srcMat[0].length;

    // Compute mean of src and dst.
    final srcMean = srcMat.mean();
    final dstMean = dstMat.mean();

    // Subtract mean from src and dst.
    final srcDemean = srcMat.map((vector) => m2d.subtraction(vector, srcMean));
    final dstDemean = dstMat.map((vector) => m2d.subtraction(vector, dstMean));

    // Eq. (38).
    final A = (dstDemean.transpose * srcDemean) / num;

    // Eq. (39).
    var d = Vector.filled(dim, 1.0);
    if (_determinant(A) < 0) {
      d = d.set(dim - 1, -1);
    }

    var T = Matrix.identity(dim + 1);

    final svdResult = _svd(A);
    final U = svdResult['U']!;
    final S = svdResult['S']!;
    final V = svdResult['V']!;

    // Eq. (40) and (43).
    final rank = _matrixRank(A);
    if (rank == 0) {
      return T * double.nan;
    } else if (rank == dim - 1) {
      if (_determinant(U) * _determinant(V) > 0) {
        final uv = U * V;
        final uvAddedColumn =
            uv.insertColumns(uv.columnCount, [Vector.zero(uv.rowCount)]);
        final lastRow = List<double>.filled(uv.columnCount, 0);
        lastRow[lastRow.length - 1] = 1;
        T = uvAddedColumn.appendRow(lastRow);
      } else {
        final s = d[dim - 1];
        d = d.setValue(dim -1, -1);
        d.set
        d[dim - 1] = -1;
        T.setSubMatrix(0, 0, U * Matrix.diagonal(d) * V);
        d[dim - 1] = s;
      }
    } else {
      T.setSubMatrix(0, 0, U * Matrix.diagonal(d) * V);
    }

    var scale = 1.0;
    if (estimateScale) {
      // Eq. (41) and (42).
      scale = 1.0 /
          srcDemean
              .flatten()
              .toList()
              .map((x) => pow(x, 2))
              .reduce((a, b) => a + b) *
          (S.dot(d));
    }

    T.setSubMatrix(0, 0, T.getSubMatrix(0, 0, dim, dim) * scale);
    T.setSubColumn(
        dim, 0, dstMean - (T.getSubMatrix(0, 0, dim, dim) * srcMean) * scale);

    return T;
  }

  static double _determinant(Matrix m) {
    final int length = m.rowCount;
    if (length != m.columnCount) {
      throw Exception('Matrix must be square');
    }
    if (length == 1) {
      return m[0][0];
    } else if (length == 2) {
      return m[0][0] * m[1][1] - m[0][1] * m[1][0];
    } else {
      throw Exception('Determinant for Matrix larger than 2x2 not implemented');
    }
  }

  static int _matrixRank(Matrix m) {
    final svdResult = _svd(m);
    final S = svdResult['S']!;
    final rank = S.asFlattenedList.where((element) => element > 1e-10).length;
    return rank;
  }

  /// Computes the singular value decomposition of a matrix, using https://lucidar.me/en/mathematics/singular-value-decomposition-of-a-2x2-matrix/
  static Map<String, Matrix> _svd(Matrix m) {
    if (m.rowCount != 2 || m.columnCount != 2) {
      throw Exception('Matrix must be 2x2');
    }
    final a = m[0][0];
    final b = m[0][1];
    final c = m[1][0];
    final d = m[1][1];

    // Computation of U matrix
    final tempCalc = a * a + b * b - c * c - d * d;
    final theta = 0.5 * math.atan2(2 * a * c + 2 * b * d, tempCalc);
    final U = Matrix.fromList([
      [math.cos(theta), -math.sin(theta)],
      [math.sin(theta), math.cos(theta)],
    ]);

    // Computation of S matrix
    // ignore: non_constant_identifier_names
    final S1 = a * a + b * b + c * c + d * d;
    // ignore: non_constant_identifier_names
    final S2 =
        math.sqrt(math.pow(tempCalc, 2) + 4 * math.pow(a * c + b * d, 2));
    final sigma1 = math.sqrt((S1 + S2) / 2);
    final sigma2 = math.sqrt((S1 - S2) / 2);
    final S = Matrix.fromList([
      [sigma1, 0],
      [0, sigma2],
    ]);

    // Computation of V matrix
    final tempCalc2 = a * a - b * b + c * c - d * d;
    final phi = 0.5 * math.atan2(2 * a * b + 2 * c * d, tempCalc2);
    final s11 = (a * math.cos(theta) + c * math.sin(theta)) * math.cos(phi) +
        (b * math.cos(theta) + d * math.sin(theta)) * math.sin(phi);
    final s22 = (a * math.sin(theta) - c * math.cos(theta)) * math.sin(phi) +
        (-b * math.sin(theta) + d * math.cos(theta)) * math.cos(phi);
    final V = Matrix.fromList([
      [s11.sign * math.cos(phi), -s22.sign * math.sin(phi)],
      [s11.sign * math.sin(phi), s22.sign * math.cos(phi)],
    ]);

    return {
      'U': U,
      'S': S,
      'V': V,
    };
  }

}

extension SetVectorValues on Vector {
  Vector setValues(int start, int end, Iterable<double> values) {
    if (values.length > length) {
      throw Exception('Values cannot be larger than vector');
    } else if (end - start != values.length) {
      throw Exception('Values must be same length as range');
    } else if (start < 0 || end > length) {
      throw Exception('Range must be within vector');
    }
    final tempList = toList();
    tempList.replaceRange(start, end, values);
    final newVector = Vector.fromList(tempList);
    return newVector;
  }

  Vector setValue(int index, double value) {
    if (index < 0 || index > length) {
      throw Exception('Index must be within range of vector');
    }
    final tempList = toList();
    tempList[index] = value;
    final newVector = Vector.fromList(tempList);
    return newVector;
  }
}

extension MatrixMean on Matrix {
  List<double> mean() {
    final meanList = <double>[];
    for (var i = 0; i < length; i++) {
      final column = this[i];
      final columnMean = column.sum() / column.length;
      meanList.add(columnMean);
    }
    return meanList;
  }
}
