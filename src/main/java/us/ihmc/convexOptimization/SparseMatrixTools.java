package us.ihmc.convexOptimization;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.sparse.csc.CommonOps_DSCC;

import static org.ejml.UtilEjml.stringShapes;

public class SparseMatrixTools
{
   public static void insert(DMatrixSparseCSC src, DMatrixSparseCSC dest, int destY0, int destX0)
   {
      CommonOps_DSCC.extract(src, 0, src.getNumRows(), 0, src.getNumCols(), dest, destY0, destX0);
   }
}
