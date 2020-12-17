package us.ihmc.convexOptimization;

import org.ejml.EjmlUnitTests;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.sparse.csc.RandomMatrices_DSCC;
import org.junit.jupiter.api.Test;
import us.ihmc.commons.RandomNumbers;
import us.ihmc.matrixlib.MatrixTestTools;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SparseMatrixToolsTest
{
   @Test
   public void testVerticallyStackMatrices()
   {
      Random random = new Random(1738L);
      for (int iter = 0; iter < 50; iter++)
      {
         int numCols = RandomNumbers.nextInt(random, 2, 50);
         int numSrcs = RandomNumbers.nextInt(random, 1, 10);
         int totalRows = 0;
         ArrayList<DMatrixSparseCSC> srcs = new ArrayList<>();
         for (int i = 0; i < numSrcs; i++)
         {
            int rows = RandomNumbers.nextInt(random, 2, 20);
            srcs.add(RandomMatrices_DSCC.rectangle(rows, numCols, (rows - 1) * (numCols - 1), random));
            totalRows += rows;
         }

         DMatrixSparseCSC destExpected = new DMatrixSparseCSC(totalRows, numCols);
         DMatrixSparseCSC dest = new DMatrixSparseCSC(totalRows, numCols);

         SparseMatrixTools.verticallyStackMatrices(srcs, dest);

         int rowOffset = 0;
         for (int i = 0; i < numSrcs; i++)
         {
            DMatrixSparseCSC src = srcs.get(i);
            SparseMatrixTools.insert(src, destExpected, rowOffset, 0);

            rowOffset += src.getNumRows();
         }

         MatrixTestTools.assertMatrixEquals(destExpected, dest, 1e-7);
      }
   }

   @Test
   public void testSimpleVerticallyStackMatrices()
   {
      DMatrixSparseCSC src1 = new DMatrixSparseCSC(2, 4);
      src1.set(0, 0, 1.0);
      src1.set(0, 1, 1.0);
      src1.set(1, 2, 1.0);
      src1.set(1, 3, 1.0);

      List<DMatrixSparseCSC> src = new ArrayList<>();
      src.add(src1);
      src.add(src1);

      DMatrixSparseCSC destExpected = new DMatrixSparseCSC(4, 4);
      DMatrixSparseCSC dest = new DMatrixSparseCSC(4, 4);
      destExpected.set(0, 0, 1.0);
      destExpected.set(0, 1, 1.0);
      destExpected.set(1, 2, 1.0);
      destExpected.set(1, 3, 1.0);
      destExpected.set(2, 0, 1.0);
      destExpected.set(2, 1, 1.0);
      destExpected.set(3, 2, 1.0);
      destExpected.set(3, 3, 1.0);

      SparseMatrixTools.verticallyStackMatrices(src, dest);

      MatrixTestTools.assertMatrixEquals(destExpected, dest, 1e-7);
   }
}
