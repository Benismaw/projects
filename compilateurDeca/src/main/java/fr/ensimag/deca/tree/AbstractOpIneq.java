package fr.ensimag.deca.tree;


import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractOpIneq extends AbstractOpCmp {

    public AbstractOpIneq(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }


    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {
        // On vérifie chaque operande
        Type typeLeft = getLeftOperand().verifyExpr(compiler, localEnv, currentClass);
        Type typeRight = getRightOperand().verifyExpr(compiler, localEnv, currentClass);

        // On verifie que ce sont des nombres
        if ((!typeLeft.isInt() && !typeLeft.isFloat()) || (!typeRight.isInt() && !typeRight.isFloat())) {
            throw new ContextualError("Les inégalités demandent des nombres (int ou float). Types obtenus : " +
                    typeLeft + " et " + typeRight + " (règle 3.33)", getLocation());
        }

        // S'il y a un float il faut convertir l'autre operande en float
        if (typeRight.isFloat() || typeLeft.isFloat()) {
            if  (typeRight.isInt()) {
                ConvFloat conv = new ConvFloat(getRightOperand());
                conv.verifyExpr(compiler, localEnv, currentClass);
                setRightOperand(conv);
            }
            else if (typeLeft.isInt()) {
                ConvFloat conv = new ConvFloat(getLeftOperand());
                conv.verifyExpr(compiler, localEnv, currentClass);
                setLeftOperand(conv);
            }
        }

        setType(compiler.environmentType.BOOLEAN);
        return compiler.environmentType.BOOLEAN;
    }
}
