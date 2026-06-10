package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMB;
import fr.ensimag.ARM.Instructions.ARMBlt;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.SLT;
/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Lower extends AbstractOpIneq {

    public Lower(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        compiler.addInstruction(new CMP(nextRegister, register));
        compiler.addInstruction(new SLT(register));
    }


    @Override
    protected String getOperatorName() {
        return "<";
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();

        String labelTrue = "lt_true_" + this.hashCode();
        String labelEnd = "lt_end_" + this.hashCode();

        getLeftOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMPush(register));

        getRightOperand().codeGenExprARM(compiler, ARMRegister.R0);
        prog.addInstruction(new ARMPop(ARMRegister.R1));

        prog.addInstruction(new ARMCmp(ARMRegister.R1, ARMRegister.R0));

        prog.addInstruction(new ARMBlt(labelTrue));

        prog.addInstruction(new ARMMov(ARMRegister.R0, new ARMImmediate(0)));
        prog.addInstruction(new ARMB(labelEnd));

        prog.addInstruction(new ARMLabel(labelTrue));
        prog.addInstruction(new ARMMov(register, new ARMImmediate(1)));

        prog.addInstruction(new ARMLabel(labelEnd));
    }

}
