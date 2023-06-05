import 'dart:math' as math;
import 'dart:developer' as devtools show log;

import 'package:scidart/numdart.dart';

void main () {
  const double score = -3.0;

  final double convertedWithScidart = 1.0 / (1.0 + exp(-score));
  final double convertedWithMath = 1.0 / (1.0 + math.exp(-score));

  print('Scidart: $convertedWithScidart');
  print('Math: $convertedWithMath');
  }