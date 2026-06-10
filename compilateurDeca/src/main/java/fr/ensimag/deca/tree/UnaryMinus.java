package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.Instructions.ARMReverseSub;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.instructions.OPP;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class UnaryMinus extends AbstractUnaryExpr {

    public UnaryMinus(AbstractExpr operand) {
        super(operand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        Type type = getOperand().verifyExpr(compiler, localEnv, currentClass);
        if (!type.isInt() && !type.isFloat()) {
            throw new ContextualError("L'opérande de '-' unaire doit être int ou float. Type trouvé : " + type + " (règle 3.37)", getLocation());
        }

        setType(type);
        return type;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        getOperand().codeGenExpr(compiler, register);
        compiler.addInstruction(new OPP(register,register));
    }


    @Override
    protected String getOperatorName() {
        return "-";
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        getOperand().codeGenExprARM(compiler, register);

        prog.addInstruction(new ARMReverseSub(register, register));

    }

}
