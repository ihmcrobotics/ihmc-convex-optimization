package us.ihmc.convexOptimization;

import gnu.trove.list.array.TIntArrayList;
import org.ejml.EjmlUnitTests;
import org.ejml.data.DMatrixSparseCSC;
import org.ejml.sparse.csc.RandomMatrices_DSCC;
import org.junit.jupiter.api.Test;
import us.ihmc.commons.RandomNumbers;
import us.ihmc.log.LogTools;
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
   public void testVerticallyStackMatricesWithColumns()
   {
      Random random = new Random(1738L);
      for (int iter = 0; iter < 50; iter++)
      {
         int numColSrcs = RandomNumbers.nextInt(random, 1, 10);
         int numRowSrcs = RandomNumbers.nextInt(random, 1, 10);

         int totalRows = 0;

         ArrayList<ArrayList<DMatrixSparseCSC>> srcs = new ArrayList<>();
         TIntArrayList colSize = new TIntArrayList();

         for (int col = 0; col < numColSrcs; col++)
         {
            srcs.add(new ArrayList<>());
            colSize.add(RandomNumbers.nextInt(random, 2, 50));

         }

         for (int row = 0; row < numRowSrcs; row++)
         {
            int numRows = RandomNumbers.nextInt(random, 2, 50);

            for (int col = 0; col < numColSrcs; col++)
            {
               int numCols = colSize.get(col);

               srcs.get(col).add(RandomMatrices_DSCC.rectangle(numRows, numCols, (numRows - 1) * (numCols - 1), random));
            }
            totalRows += numRows;
         }
         int totalCols = colSize.sum();

         DMatrixSparseCSC destExpected = new DMatrixSparseCSC(totalRows, totalCols);
         DMatrixSparseCSC dest = new DMatrixSparseCSC(totalRows, totalCols);

         int colStart = 0;
         for (int col = 0; col < numColSrcs; col++)
         {
            SparseMatrixTools.verticallyStackMatrices(srcs.get(col), dest, colStart);
            colStart += colSize.get(col);
         }

         int colOffset = 0;
         for (int col = 0; col < numColSrcs; col++)
         {
            int rowOffset = 0;
            for (int row = 0; row < numRowSrcs; row++)
            {
               DMatrixSparseCSC src = srcs.get(col).get(row);
               SparseMatrixTools.insert(src, destExpected, rowOffset, colOffset);

               rowOffset += src.getNumRows();
            }
            colOffset += colSize.get(col);
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
