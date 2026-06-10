package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.instructions.FLOAT;

/**
 * Conversion of an int into a float. Used for implicit conversions.
 * 
 * @author gl16
 * @date 01/01/2026
 */
public class ConvFloat extends AbstractUnaryExpr {
    public ConvFloat(AbstractExpr operand) {
        super(operand);
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        // L'expression a deja etait verifié avant
        //getOperand().verifyExpr(compiler, localEnv, currentClass);

        setType(compiler.environmentType.FLOAT);
        return compiler.environmentType.FLOAT;
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        getOperand().codeGenExpr(compiler, register);
        compiler.addInstruction(new FLOAT(register,register));
    }


    @Override
    protected String getOperatorName() {
        return "/* conv float */";
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
    }     

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        getOperand().codeGenExprARM(compiler, register);

        prog.addRaw("""
                vmov s0, r0
                vcvt.f32.s32 s0, s0
        """);
    }
    

}
