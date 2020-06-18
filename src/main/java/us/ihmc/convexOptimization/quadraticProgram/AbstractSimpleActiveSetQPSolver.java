package us.ihmc.convexOptimization.quadraticProgram;

import org.ejml.data.DMatrixRMaj;

public abstract class AbstractSimpleActiveSetQPSolver implements ActiveSetQPSolver
{
   protected final DMatrixRMaj quadraticCostQMatrix = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj quadraticCostQVector = new DMatrixRMaj(0, 0);
   protected double quadraticCostScalar;

   protected final DMatrixRMaj linearEqualityConstraintsAMatrix = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj linearEqualityConstraintsBVector = new DMatrixRMaj(0, 0);

   protected final DMatrixRMaj linearInequalityConstraintsCMatrixO = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj linearInequalityConstraintsDVectorO = new DMatrixRMaj(0, 0);

   protected final DMatrixRMaj variableLowerBounds = new DMatrixRMaj(0, 0);
   protected final DMatrixRMaj variableUpperBounds = new DMatrixRMaj(0, 0);
}
