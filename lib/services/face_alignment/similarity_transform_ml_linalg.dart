import 'dart:math' as math;

import 'package:matrix2d/matrix2d.dart';
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
        T = T.setValues(0, dim, 0, dim, U * V);
        // final uv = U * V;
        // final uvAddedColumn =
        //     uv.insertColumns(uv.columnCount, [Vector.zero(uv.rowCount)]);
        // final lastRow = List<double>.filled(uv.columnCount, 0);
        // lastRow[lastRow.length - 1] = 1;
        // T = uvAddedColumn.appendRow(lastRow);
      } else {
        final s = d[dim - 1];
        d = d.set(dim - 1, -1);
        T = T.setValues(0, dim, 0, dim, U * Matrix.diagonal(d.toList()));
        d = d.set(dim - 1, s);
      }
    } else {
      T = T.setValues(0, dim, 0, dim, U * Matrix.diagonal(d.toList()));
    }

    var scale = 1.0;
    if (estimateScale) {
      // Eq. (41) and (42).
      final srcDemeanVarSum = srcDemean.variance(Axis.columns).sum();
      final Sd = (S * d).asFlattenedList.sum();
      scale = 1.0 / srcDemeanVarSum * Sd;
    }

    final subTIndices = Iterable<int>.generate(dim, (index) => index);
    final subT = T.sample(rowIndices: subTIndices, columnIndices: subTIndices);
    final newSubT = dstMean - (subT * srcMean) * scale;
    T = T.setValues(0, dim, 0, dim, Matrix.fromColumns([newSubT]));
    final newNewSubT = T.sample(rowIndices: subTIndices, columnIndices: subTIndices) * scale;
    T = T.setValues(0, dim, 0, dim, newNewSubT);

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

  // Vector setValue(int index, double value) {
  //   if (index < 0 || index > length) {
  //     throw Exception('Index must be within range of vector');
  //   }
  //   final tempList = toList();
  //   tempList[index] = value;
  //   final newVector = Vector.fromList(tempList);
  //   return newVector;
  // }
}

extension ChangeMatrixValues on Matrix {
  Matrix setValues(
    int startRow,
    int endRow,
    int startColumn,
    int endColumn,
    Iterable<Iterable<double>> values,
  ) {
    if (values.length > rowCount) {
      throw Exception('New values cannot have more rows than original matrix');
    } else if (values.elementAt(0).length > columnCount) {
      throw Exception(
          'New values cannot have more columns than original matrix');
    } else if (endRow - startRow != values.length) {
      throw Exception('Values (number of rows) must be same length as range');
    } else if (endColumn - startColumn != values.elementAt(0).length) {
      throw Exception(
          'Values (number of columns) must be same length as range');
    } else if (startRow < 0 ||
        endRow > rowCount ||
        startColumn < 0 ||
        endColumn > columnCount) {
      throw Exception('Range must be within matrix');
    }
    final tempList = asFlattenedList;
    for (var i = startRow; i < endRow; i++) {
      tempList.replaceRange(i * columnCount + startColumn,
          i * columnCount + endColumn, values.elementAt(i).toList());
    }
    final newMatrix = Matrix.fromFlattenedList(tempList, rowCount, columnCount);
    return newMatrix;
  }

  Matrix setValue(int row, int column, double value) {
    if (row < 0 || row > rowCount || column < 0 || column > columnCount) {
      throw Exception('Index must be within range of matrix');
    }
    final tempList = asFlattenedList;
    tempList[row * columnCount + column] = value;
    final newMatrix = Matrix.fromFlattenedList(tempList, rowCount, columnCount);
    return newMatrix;
  }

  Matrix appendRow(List<double> row) {
    final oldNumberOfRows = rowCount;
    final oldNumberOfColumns = columnCount;
    if (row.length != oldNumberOfColumns) {
      throw Exception('Row must have same number of columns as matrix');
    }
    final flatListMatrix = asFlattenedList;
    flatListMatrix.addAll(row);
    return Matrix.fromFlattenedList(
      flatListMatrix,
      oldNumberOfRows + 1,
      oldNumberOfColumns,
    );
  }
}
