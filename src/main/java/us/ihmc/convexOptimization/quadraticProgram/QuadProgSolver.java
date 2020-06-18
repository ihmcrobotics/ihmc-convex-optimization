package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;

import us.ihmc.convexOptimization.QuadProgWrapper;
import us.ihmc.convexOptimization.exceptions.NoConvergenceException;

public class QuadProgSolver extends ConstrainedQPSolver
{

   /*
    * The problem is in the form: min 0.5 * x G x + g0 x s.t. CE^T x + ce0 = 0 CI^T x + ci0 >= 0
    */

   QuadProgWrapper qpWrapper;
   DMatrixRMaj negAin = new DMatrixRMaj(0), negAeq = new DMatrixRMaj(0);

   public QuadProgSolver()
   {
      this(1, 0, 0);
   }

   public QuadProgSolver(int nvar, int neq, int nin)
   {
      qpWrapper = new QuadProgWrapper(nvar, neq, nin);
      allocateTempraryMatrixOnDemand(nvar, neq, nin);

   }

   private void allocateTempraryMatrixOnDemand(int nvar, int neq, int nin)
   {
      negAin.reshape(nvar, nin);
      negAeq.reshape(nvar, neq);
   }

   @Override
   public boolean supportBoxConstraints()
   {
      return false;
   }

   @Override
   public int solve(DMatrixRMaj Q, DMatrixRMaj f, DMatrixRMaj Aeq, DMatrixRMaj beq, DMatrixRMaj Ain, DMatrixRMaj bin, DMatrixRMaj x,
                    boolean initialize)
         throws NoConvergenceException
   {
      return solve(Q, f, Aeq, beq, Ain, bin, null, null, x, initialize);
   }

   @Override
   public int solve(DMatrixRMaj Q, DMatrixRMaj f, DMatrixRMaj Aeq, DMatrixRMaj beq, DMatrixRMaj Ain, DMatrixRMaj bin, DMatrixRMaj lb,
                    DMatrixRMaj ub, DMatrixRMaj x, boolean initialize)
         throws NoConvergenceException
   {

      allocateTempraryMatrixOnDemand(Q.numCols, Aeq.numRows, Ain.numRows);

      CommonOps_DDRM.transpose(Aeq, negAeq);
      CommonOps_DDRM.scale(-1, negAeq);
      CommonOps_DDRM.transpose(Ain, negAin);
      CommonOps_DDRM.scale(-1, negAin);
      return qpWrapper.solve(Q, f, negAeq, beq, negAin, bin, x, initialize);
   }

   public double getCost()
   {
      return qpWrapper.getObjVal();
   }
}
