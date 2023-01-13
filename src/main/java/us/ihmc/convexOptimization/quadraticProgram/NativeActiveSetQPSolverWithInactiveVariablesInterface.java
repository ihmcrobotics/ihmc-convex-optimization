package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixRMaj;
import us.ihmc.matrixlib.NativeMatrix;

/**
 * Extension of a {@link ActiveSetQPSolver} allowing to explicitly disable or enable components of
 * {@code x} when solving the problem.
 * <p>
 * Instead of resizing the problem's matrices, the user can simply indicate which component is to be
 * ignore during the next {@link #solve(DMatrixRMaj)}, cutting computation time to a similar
 * magnitude compared to the problem being resized in a compact form. This has the advantage from
 * the user's perspective to preserve a fixed indexing for each variable component without
 * increasing computation time.
 * </p>
 */
public interface NativeActiveSetQPSolverWithInactiveVariablesInterface extends ActiveSetQPSolverWithInactiveVariablesInterface
{
   NativeMatrix getCostHessianUnsafe();

   NativeMatrix getCostGradientUnsafe();

   NativeMatrix getAeqUnsafe();

   NativeMatrix getBeqUnsafe();

   NativeMatrix getAinUnsafe();

   NativeMatrix getBinUnsafe();

   NativeMatrix getLowerBoundsUnsafe();

   NativeMatrix getUpperBoundsUnsafe();
}
