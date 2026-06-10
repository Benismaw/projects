package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMBeq;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMDiv;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMMul;
import fr.ensimag.ARM.Instructions.ARMSub;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.REM;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Modulo extends AbstractOpArith {

    public Modulo(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }


    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        Type rightType = getRightOperand().verifyExpr(compiler, localEnv, currentClass);
        Type leftType = getLeftOperand().verifyExpr(compiler, localEnv, currentClass);

        if (!rightType.isInt() || !leftType.isInt()) {
            throw new ContextualError("Les opérandes du modulo doivent être de type int (règle 3.33)", getLocation());
        }

        setType(compiler.environmentType.INT);
        return compiler.environmentType.INT;
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {

        boolean noCheck = compiler.getCompilerOptions().getNoCheck();

        if (!noCheck) {
            compiler.addInstruction(new CMP(new ImmediateInteger(0), nextRegister));
            compiler.addInstruction(new BEQ(new Label("division_zero")));
        }
        compiler.addInstruction(new REM(nextRegister, register));
    }

    @Override
    protected String getOperatorName() {
        return "%";
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        getRightOperand().codeGenExpr(compiler, register);

        if (getLeftOperand() instanceof Identifier) {
            Identifier left = (Identifier) getLeftOperand();
            fr.ensimag.ima.pseudocode.DAddr addr = left.getExpDefinition().getOperand();

            compiler.addInstruction(new LOAD(addr, Register.R1));
            compiler.addInstruction(new REM(register, Register.R1));
            
            if (!compiler.getCompilerOptions().getNoCheck()) {
                compiler.addInstruction(new BOV(new Label("division_zero")));
            }
            
            compiler.addInstruction(new LOAD(Register.R1, register));
        } else {
            super.codeGenExpr(compiler, register);
        }
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        getLeftOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMMov(ARMRegister.R2, ARMRegister.R0));

        getRightOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMMov(ARMRegister.R3, ARMRegister.R0));

        if (!compiler.getCompilerOptions().getNoCheck()) {
            prog.addInstruction(new ARMMov(ARMRegister.R4, new ARMImmediate(0)));
            prog.addInstruction(new ARMCmp(ARMRegister.R3, ARMRegister.R4));
            prog.addInstruction(new ARMBeq("division_zero"));
        }

        prog.addInstruction(new ARMMov(ARMRegister.R0, ARMRegister.R3));
        prog.addInstruction(new ARMMov(ARMRegister.R1, ARMRegister.R2));
        prog.addInstruction(new ARMDiv());

        prog.addInstruction(new ARMMul(ARMRegister.R0, ARMRegister.R0, ARMRegister.R3));
        prog.addInstruction(new ARMSub(ARMRegister.R0, ARMRegister.R2, ARMRegister.R0));
    }


}
