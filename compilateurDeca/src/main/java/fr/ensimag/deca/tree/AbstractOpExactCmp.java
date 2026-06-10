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
public abstract class AbstractOpExactCmp extends AbstractOpCmp {

    public AbstractOpExactCmp(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
                           ClassDefinition currentClass) throws ContextualError {
        // On vérifie chaque operande
        Type typeLeft = getLeftOperand().verifyExpr(compiler, localEnv, currentClass);
        Type typeRight = getRightOperand().verifyExpr(compiler, localEnv, currentClass);

        boolean compatible = false;
        // On verifie si ce sont des nombres
        if ((typeLeft.isInt() || typeLeft.isFloat()) && (typeRight.isInt() || typeRight.isFloat())) {
            compatible = true;

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
        }
        else if (typeLeft.isBoolean() && typeRight.isBoolean()) {
            compatible = true;
        }

        else if (typeLeft.isClassOrNull() && typeRight.isClassOrNull()) {
            compatible = true;
        }

        if (!compatible) {
            throw new ContextualError("Comparaison invalide entre " + typeLeft + " et " + typeRight + " (règle 3.33)", getLocation());
        }

        setType(compiler.environmentType.BOOLEAN);
        return compiler.environmentType.BOOLEAN;
    }


}
