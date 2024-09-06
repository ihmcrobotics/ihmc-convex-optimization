package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import us.ihmc.convexOptimization.ReLUQPWrapper;
import us.ihmc.convexOptimization.exceptions.NoConvergenceException;

public class ReLUQPSolver // TODO: extend ConstrainedQPSolver
{
   ReLUQPWrapper qpWrapper;

   public ReLUQPSolver()
   {
      qpWrapper = new ReLUQPWrapper();
   }

   // TODO: Solve function is standardized in ConstrainedQPSolver and I should be using those inputs but
   //  I'm hacking this together right now. If extending ConstrainedQPSolver and using standardized inputs
   //  then this setup function would be doing more than just calling the wrapper immediately.
   public void setup(DMatrixRMaj H, DMatrixRMaj g, DMatrixRMaj A, DMatrixRMaj lb, DMatrixRMaj ub,
                     boolean verbose, double epsPrimal, double epsDual, int maxIters, int itersBetweenChecks)
   {
      qpWrapper.setup(H, g, A, lb, ub, verbose, epsPrimal, epsDual, maxIters, itersBetweenChecks);
   }

   public int update(DMatrixRMaj gNew, DMatrixRMaj lbNew, DMatrixRMaj ubNew)
   {
      return qpWrapper.update(gNew, lbNew, ubNew);
   }

   public int solve()
   {
      return qpWrapper.solve();
   }

   public DMatrixRMaj getSolution()
   {
      return qpWrapper.getSolution();
   }
}
