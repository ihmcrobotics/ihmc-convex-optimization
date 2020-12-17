package us.ihmc.convexOptimization;

import org.ejml.MatrixDimensionException;
import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.sparse.csc.CommonOps_DSCC;
import us.ihmc.matrixlib.NativeMatrix;

import java.util.List;

import static org.ejml.UtilEjml.stringShapes;

public class SparseMatrixTools
{
   public static void insert(DMatrixSparseCSC src, DMatrixSparseCSC dest, int destY0, int destX0)
   {
      CommonOps_DSCC.extract(src, 0, src.getNumRows(), 0, src.getNumCols(), dest, destY0, destX0);
   }

   public static void verticallyStackMatrices(List<DMatrixSparseCSC> srcs, DMatrixSparseCSC dest)
   {
      int totalRows = 0;
      int totalNz = 0;
      for (int i = 0; i < srcs.size(); i++)
      {
         DMatrixSparseCSC src = srcs.get(i);
         if (src.getNumCols() != dest.getNumCols())
            throw new IllegalArgumentException("Number of cols do not match. " + src.getNumCols() + " != " + dest.getNumCols());

         totalRows += src.getNumRows();
         totalNz += src.getNonZeroLength();
      }

      if (totalRows != dest.getNumRows())
         throw new IllegalArgumentException("Number of rows do not match. " + totalRows + " != " + dest.getNumCols());

      dest.growMaxLength(totalNz, false);

      int destValIdx = 0;
      for (int col = 0; col < dest.getNumCols(); col++)
      {
         int rowOffset = 0;
         dest.col_idx[col] = destValIdx;

         for (int srcIdx = 0; srcIdx < srcs.size(); srcIdx++)
         {
            DMatrixSparseCSC src = srcs.get(srcIdx);

            for (int srcValIdx = src.col_idx[col]; srcValIdx < src.col_idx[col + 1]; srcValIdx++)
            {
               int srcRow = src.nz_rows[srcValIdx];
               double srcVal = src.nz_values[srcValIdx];
               dest.nz_values[destValIdx] = srcVal;
               dest.nz_rows[destValIdx] = srcRow + rowOffset;

               destValIdx++;
            }

            rowOffset += src.getNumRows();
         }
      }
      dest.col_idx[dest.getNumCols()] = destValIdx;
   }
}
