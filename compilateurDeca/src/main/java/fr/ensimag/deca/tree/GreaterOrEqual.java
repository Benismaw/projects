package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMB;
import fr.ensimag.ARM.Instructions.ARMBge;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.SGE;

/**
 * Operator "x >= y"
 * 
 * @author gl16
 * @date 01/01/2026
 */
public class GreaterOrEqual extends AbstractOpIneq {

    public GreaterOrEqual(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        compiler.addInstruction(new CMP(nextRegister, register));
        compiler.addInstruction(new SGE(register));
    }


    @Override
    protected String getOperatorName() {
        return ">=";
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        String labelTrue = "ge_true_" + this.hashCode();
        String labelEnd = "ge_end_" + this.hashCode();

        getLeftOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMPush(ARMRegister.R0));

        getRightOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMPop(ARMRegister.R1));

        prog.addInstruction(new ARMCmp(ARMRegister.R1, ARMRegister.R0));

        prog.addInstruction(new ARMBge(labelTrue));

        prog.addInstruction(new ARMMov(ARMRegister.R0, new ARMImmediate(0)));
        prog.addInstruction(new ARMB(labelEnd));

        prog.addInstruction(new ARMLabel(labelTrue));
        prog.addInstruction(new ARMMov(ARMRegister.R0, new ARMImmediate(1)));

        prog.addInstruction(new ARMLabel(labelEnd));
    }
}
