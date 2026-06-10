package fr.ensimag.deca.tree;

import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;

/**
 * Arithmetic binary operations (+, -, /, ...)
 * 
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractOpArith extends AbstractBinaryExpr {

    public AbstractOpArith(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {

        Type typeLeft = getLeftOperand().verifyExpr(compiler, localEnv, currentClass);
        Type typeRight = getRightOperand().verifyExpr(compiler, localEnv, currentClass);

        // Règle 3.33 : Les opérandes doivent être int ou float
        if (!typeLeft.isInt() && !typeLeft.isFloat()) {
            throw new ContextualError("L'opérande gauche d'une opération arithmétique doit être int ou float, mais est " + typeLeft + " (règle 3.33)", getLocation());
        }
        if (!typeRight.isInt() && !typeRight.isFloat()) {
            throw new ContextualError("L'opérande droite d'une opération arithmétique doit être int ou float, mais est " + typeRight + " (règle 3.33)", getLocation());
        }

        Type typeResult;
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
            typeResult = compiler.environmentType.FLOAT;
        }
        else {
            typeResult = compiler.environmentType.INT;
        }

        setType(typeResult);
        return typeResult;
    }
}
