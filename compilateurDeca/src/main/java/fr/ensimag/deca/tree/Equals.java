package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMB;
import fr.ensimag.ARM.Instructions.ARMBeq;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.SEQ;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Equals extends AbstractOpExactCmp {

    public Equals(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        compiler.addInstruction(new CMP(nextRegister, register));
        compiler.addInstruction(new SEQ(register));
    }


    @Override
    protected String getOperatorName() {
        return "==";
    }    
    
    @Override
protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {

    var prog = compiler.getARMProgram();

    
    if (getLeftOperand().getType().isFloat()
        || getRightOperand().getType().isFloat()) {

        
        getLeftOperand().codeGenExprARM(compiler, ARMRegister.R0);

        if (getLeftOperand().getType().isInt()) {
            prog.addRaw("""
                vmov s1, r0
                vcvt.f32.s32 s1, s1
            """);
        } else {
            prog.addRaw("""
                vmov s1, r0
            """);
        }

        
        getRightOperand().codeGenExprARM(compiler, ARMRegister.R0);

        if (getRightOperand().getType().isInt()) {
            prog.addRaw("""
                vmov s0, r0
                vcvt.f32.s32 s0, s0
            """);
        } else {
            prog.addRaw("""
                vmov s0, r0
            """);
        }

        String lTrue = "eq_true_" + hashCode();
        String lEnd  = "eq_end_" + hashCode();

        prog.addRaw("""
            vcmp.f32 s1, s0
            vmrs APSR_nzcv, fpscr
        """);

        prog.addInstruction(new ARMBeq(lTrue));
        prog.addInstruction(new ARMMov(register, new ARMImmediate(0)));
        prog.addInstruction(new ARMB(lEnd));
        prog.addInstruction(new ARMLabel(lTrue));
        prog.addInstruction(new ARMMov(register, new ARMImmediate(1)));
        prog.addInstruction(new ARMLabel(lEnd));
        return;
    }

    
    getLeftOperand().codeGenExprARM(compiler, ARMRegister.R0);
    prog.addInstruction(new ARMPush(ARMRegister.R0));
    getRightOperand().codeGenExprARM(compiler, ARMRegister.R0);
    prog.addInstruction(new ARMPop(ARMRegister.R1));
    prog.addInstruction(new ARMCmp(ARMRegister.R1, ARMRegister.R0));
}

    
}
