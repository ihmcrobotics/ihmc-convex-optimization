package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.factory.LinearSolverFactory_DDRM;
import org.ejml.interfaces.linsol.LinearSolverDense;

public class BlockDiagSquareMatrix extends DMatrixRMaj
{
   private static final long serialVersionUID = 8813856249678942997L;

   int[] blockSizes;
   int[] blockStarts;

   DMatrixRMaj[] tmpMatrix;

   public BlockDiagSquareMatrix(int... blockSizes)
   {
      super(0);
      this.blockSizes = blockSizes;
      blockStarts = new int[getNumBlocks() + 1];
      tmpMatrix = new DMatrixRMaj[getNumBlocks()];
      int matrixRows = 0;
      for (int i = 0; i < getNumBlocks(); i++)
      {
         tmpMatrix[i] = new DMatrixRMaj(blockSizes[i], blockSizes[i]);
         blockStarts[i] = matrixRows;
         matrixRows += blockSizes[i];
      }

      blockStarts[blockStarts.length - 1] = matrixRows;
      super.reshape(matrixRows, matrixRows);
   }

   public int getNumBlocks()
   {
      return blockSizes.length;
   }

   public void setBlock(DMatrixRMaj srcBlock, int blockId)
   {
      setBlock(srcBlock, blockId, this);
   }

   public void setBlock(DMatrixRMaj srcBlock, int blockId, DMatrixRMaj dstMatrix)
   {
      dstMatrix.reshape(numRows, numCols);
      int startIndex = blockStarts[blockId];
      CommonOps_DDRM.insert(srcBlock, dstMatrix, startIndex, startIndex);
   }

   public void packBlock(DMatrixRMaj dstBlock, int blockId, int destX0, int destY0)
   {
      int startIndex = blockStarts[blockId];
      int endIndex = blockStarts[blockId + 1];
      CommonOps_DDRM.extract(this, startIndex, endIndex, startIndex, endIndex, dstBlock, destX0, destY0);
   }

   public void packInverse(LinearSolverDense<DMatrixRMaj> solver, BlockDiagSquareMatrix matrixToPack)
   {
      for (int i = 0; i < blockSizes.length; i++)
      {
         tmpMatrix[i].reshape(blockSizes[i], blockSizes[i]);
         packBlock(tmpMatrix[i], i, 0, 0);
         solver.setA(tmpMatrix[i]);
         solver.invert(tmpMatrix[i]);
         matrixToPack.setBlock(tmpMatrix[i], i);
      }
   }

   public void packInverse(LinearSolverDense<DMatrixRMaj> solver, DMatrixRMaj matrixToPack)
   {
      matrixToPack.zero();
      for (int i = 0; i < blockSizes.length; i++)
      {
         tmpMatrix[i].reshape(blockSizes[i], blockSizes[i]);
         packBlock(tmpMatrix[i], i, 0, 0);
         solver.setA(tmpMatrix[i]);
         solver.invert(tmpMatrix[i]);
         setBlock(tmpMatrix[i], i, matrixToPack);
      }
   }

   /**
    * c = this*b<sup>T</sup>
    * 
    * @param b
    * @param c
    */
   DMatrixRMaj multTempB = new DMatrixRMaj(0);
   DMatrixRMaj multTempC = new DMatrixRMaj(0);

   public void multTransB(DMatrixRMaj b, DMatrixRMaj c)
   {
      for (int i = 0; i < blockSizes.length; i++)
      {
         for (int crow = blockStarts[i]; crow < blockStarts[i + 1]; crow++)
         {
            int aIndex0 = getIndex(crow, blockStarts[i]);
            for (int ccol = 0; ccol < c.numCols; ccol++)
            {
               double val = 0.0;
               int aIndex = aIndex0;
               int bIndex = b.getIndex(ccol, blockStarts[i]);
               int bEnd = bIndex + blockSizes[i];
               while (bIndex < bEnd)
                  val += data[aIndex++] * b.data[bIndex++];

               c.set(crow, ccol, val);
            }
            /*
             * tmpMatrix[i].reshape(blockSizes[i] , blockSizes[i]); packBlock(tmpMatrix[i], i, 0, 0);
             * multTempB.reshape(b.numRows, blockSizes[i]); multTempC.reshape(blockSizes[i], c.numCols);
             * CommonOps_DDRM.extract(b, 0, b.numRows, blockStarts[i], blockStarts[i+1], multTempB, 0, 0);
             * CommonOps_DDRM.multTransB(tmpMatrix[i], multTempB, multTempC); CommonOps_DDRM.insert(multTempC, c,
             * blockStarts[i], 0);
             */
         }
      }

   }

   /**
    * c = this*b
    * 
    * @param b
    * @param c
    */
   public void mult(double alpha, DMatrixRMaj b, DMatrixRMaj c)
   {
      for (int i = 0; i < blockSizes.length; i++)
      {
         tmpMatrix[i].reshape(blockSizes[i], blockSizes[i]);
         packBlock(tmpMatrix[i], i, 0, 0);
         multTempB.reshape(blockSizes[i], b.numCols);
         multTempC.reshape(blockSizes[i], c.numCols);
         CommonOps_DDRM.extract(b, blockStarts[i], blockStarts[i + 1], 0, b.numCols, multTempB, 0, 0);
         CommonOps_DDRM.mult(alpha, tmpMatrix[i], multTempB, multTempC);
         CommonOps_DDRM.insert(multTempC, c, blockStarts[i], 0);
      }
   }

   public static void main(String[] arg)
   {
      BlockDiagSquareMatrix m = new BlockDiagSquareMatrix(1, 2);
      DMatrixRMaj b1 = new DMatrixRMaj(1, 1, true, 1);
      DMatrixRMaj b2 = new DMatrixRMaj(2, 2, true, 2, 3, 4, 5);

      m.setBlock(b1, 0);
      m.setBlock(b2, 1);

      System.out.println(m);

      m.packInverse(LinearSolverFactory_DDRM.general(m.numRows, m.numCols), m);
      b1.zero();
      b2.zero();

      m.packBlock(b1, 0, 0, 0);
      m.packBlock(b2, 1, 0, 0);

      System.out.println(b1);
      System.out.println(b2);
      System.out.println("m=\n" + m);
   }

}
