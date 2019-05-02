package hex.svm;

import water.MRTask;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;

public class MatrixUtils {

  static LLMatrix productMM(Frame icf, Vec diagonal) {
    Vec[] vecs = ArrayUtils.append(icf.vecs(), diagonal);
    double result[] = new ProductMMTask().doAll(vecs)._result;
    
    LLMatrix m = new LLMatrix(icf.numCols());
    int pos = 0;
    for (int i = 0; i < icf.numCols(); i++) {
      for (int j = 0; j <= i; j++) {
        m.set(i, j, result[pos++] + (i == j ? 1 : 0));
      }
    }
    return m;
  }

  private static class ProductMMTask extends MRTask<ProductMMTask> {
    // OUT
    private double[] _result;

    @Override
    public void map(Chunk[] cs) {
      final Chunk diagonal = cs[cs.length - 1];
      final int column  = cs.length - 1;
      _result = new double[(column + 1) * column / 2];
      double[] buff = new double[cs[0]._len];
      int offset = 0;
      for (int i = 0; i < column; i++) {
        offset += i;
        for (int p = 0; p < buff.length; p++) {
          buff[p] = cs[i].atd(p) * diagonal.atd(p);
        }
        for (int j = 0; j <= i; j++) {
          double tmp = 0;
          for (int p = 0; p < buff.length; p++) {
            tmp += buff[p] * cs[j].atd(p);
          }
          _result[offset+j] = tmp;
        }
      }
    }

    @Override
    public void reduce(ProductMMTask mrt) {
      ArrayUtils.add(_result, mrt._result);
    }
  }

}
