package fr.ensimag.deca.tree;

import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.ima.pseudocode.GPRegister;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractOpBool extends AbstractBinaryExpr {

    public AbstractOpBool(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {

        // On vérifie chaque operande
        Type typeLeft = getLeftOperand().verifyExpr(compiler, localEnv, currentClass);
        Type typeRight = getRightOperand().verifyExpr(compiler, localEnv, currentClass);

        // On verifie que ce sont des booléens
        if (!typeLeft.isBoolean()) {
            throw new ContextualError("L'opérande gauche d'une opération booléenne doit être boolean, mais est " + typeLeft + " (règle 3.33)", getLocation());
        }
        if (!typeRight.isBoolean()) {
            throw new ContextualError("L'opérande droite d'une opération booléenne doit être boolean, mais est " + typeRight + " (règle 3.33)", getLocation());
        }

        setType(compiler.environmentType.BOOLEAN);
        return compiler.environmentType.BOOLEAN;
    }
}
