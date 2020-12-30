package us.ihmc.convexOptimization;

import org.ejml.data.DMatrixSparseCSC;
import org.ejml.sparse.csc.CommonOps_DSCC;

import java.util.List;

public class SparseMatrixTools
{
   public static void insert(DMatrixSparseCSC src, DMatrixSparseCSC dest, int destY0, int destX0)
   {
      CommonOps_DSCC.extract(src, 0, src.getNumRows(), 0, src.getNumCols(), dest, destY0, destX0);
   }

   public static void verticallyStackMatrices(List<DMatrixSparseCSC> srcs, DMatrixSparseCSC dest)
   {
      verticallyStackMatrices(srcs, dest, 0);
   }

   public static void verticallyStackMatrices(List<DMatrixSparseCSC> srcs, DMatrixSparseCSC dest, int destStartColumn)
   {
      int totalRows = 0;
      int totalSrcNz = 0;
      int srcColumns = srcs.get(0).getNumCols();
      if (srcColumns + destStartColumn > dest.getNumCols())
         throw new IllegalArgumentException("Number of cols in destination isn't big enough.");

      for (int i = 0; i < srcs.size(); i++)
      {
         DMatrixSparseCSC src = srcs.get(i);
         if (src.getNumRows() == 0 || src.getNumCols() == 0)
            continue;

         if (src.getNumCols() != srcColumns)
            throw new IllegalArgumentException("Number of cols do not match. " + src.getNumCols() + " != " + dest.getNumCols());

         totalRows += src.getNumRows();
         totalSrcNz += src.getNonZeroLength();
      }

      if (totalRows == 0)
         return;

      if (totalRows != dest.getNumRows())
         throw new IllegalArgumentException("Number of rows do not match. " + totalRows + " != " + dest.getNumCols());

      dest.growMaxLength(dest.getNonZeroLength() + totalSrcNz, true);

      int destValIdx = dest.col_idx[destStartColumn];

      for (int srcCol = 0; srcCol < srcColumns; srcCol++)
      {
         int rowOffset = 0;
         dest.col_idx[destStartColumn + srcCol] = destValIdx;

         for (int srcIdx = 0; srcIdx < srcs.size(); srcIdx++)
         {
            DMatrixSparseCSC src = srcs.get(srcIdx);
            if (src.getNumRows() == 0 || src.getNumCols() == 0)
               continue;

            for (int srcValIdx = src.col_idx[srcCol]; srcValIdx < src.col_idx[srcCol + 1]; srcValIdx++)
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
      dest.nz_length = destValIdx;
      for (int col = srcColumns + destStartColumn; col < dest.getNumCols() + 1; col++)
         dest.col_idx[col] = destValIdx;
   }
}
