package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;

import us.ihmc.convexOptimization.QpOASESCWrapper;
import us.ihmc.convexOptimization.exceptions.NoConvergenceException;

public class OASESConstrainedQPSolver extends ConstrainedQPSolver
{

   /**
    * min 0.5*x'Hx + g'x st lbA <= Ax <= ubA lb <= x <= ub matrices are row-major
    *
    * @param nWSR    - number of working set re-calculation
    * @param cputime - maximum cputime, null
    * @param x       - initial and return variable to be optimized
    * @return returnCode from C-API
    */
   double maxCPUTime;
   double currentCPUTime;
   int maxWorkingSetChange;
   int currentWorkingSetChange;

   QpOASESCWrapper qpWrapper;

   public OASESConstrainedQPSolver()
   {
      this(1, 1);
   }

   public OASESConstrainedQPSolver(int nvar, int ncon)
   {
      maxCPUTime = Double.POSITIVE_INFINITY;
      maxWorkingSetChange = 10000;
      qpWrapper = new QpOASESCWrapper();
   }

   @Override
   public boolean supportBoxConstraints()
   {
      return true;
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

      qpWrapper.setMaxCpuTime(maxCPUTime);
      qpWrapper.setMaxWorkingSetChanges(maxWorkingSetChange);
      qpWrapper.solve(Q, f, Aeq, beq, Ain, bin, lb, ub, x, initialize);
      int iter = qpWrapper.getLastWorkingSetChanges();
      currentCPUTime = qpWrapper.getLastCpuTime();
      currentWorkingSetChange = qpWrapper.getLastWorkingSetChanges();
      return iter;
   }
}
