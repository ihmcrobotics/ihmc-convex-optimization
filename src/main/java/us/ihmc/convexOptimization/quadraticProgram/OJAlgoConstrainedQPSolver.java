package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.optimisation.ExpressionsBasedModel;
import org.ojalgo.optimisation.convex.ConvexSolver;

import us.ihmc.convexOptimization.exceptions.NoConvergenceException;

public class OJAlgoConstrainedQPSolver extends ConstrainedQPSolver
{

   public OJAlgoConstrainedQPSolver()
   {

      ExpressionsBasedModel foo;
      throw new RuntimeException("In Development... Coming soon...");
   }

   @Override
   public int solve(DMatrixRMaj Q, DMatrixRMaj f, DMatrixRMaj Aeq, DMatrixRMaj beq, DMatrixRMaj Ain, DMatrixRMaj bin, DMatrixRMaj x,
                    boolean initialize)
         throws NoConvergenceException
   {
      PrimitiveDenseStore QDenseStore = PrimitiveDenseStore.FACTORY.columns(Q.data);
      PrimitiveDenseStore CDenseStore = PrimitiveDenseStore.FACTORY.columns(f.data);

      ConvexSolver.Builder builder = new ConvexSolver.Builder(QDenseStore, CDenseStore);

      return 2;
   }

   @Override
   public int solve(DMatrixRMaj Q, DMatrixRMaj f, DMatrixRMaj Aeq, DMatrixRMaj beq, DMatrixRMaj Ain, DMatrixRMaj bin, DMatrixRMaj lb,
                    DMatrixRMaj ub, DMatrixRMaj x, boolean initialize)
         throws NoConvergenceException
   {
      return 0;
   }

   @Override
   public boolean supportBoxConstraints()
   {
      return false;
   }

}
