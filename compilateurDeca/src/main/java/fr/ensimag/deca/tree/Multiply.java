package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMMul;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.MUL;


/**
 * @author gl16
 * @date 01/01/2026
 */
public class Multiply extends AbstractOpArith {
    public Multiply(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {

        compiler.addInstruction(new MUL(nextRegister, register));

        if (getType().isFloat() && !compiler.getCompilerOptions().getNoCheck()) {
            loadErrorLine(compiler);
            compiler.addInstruction(new BOV(new Label("debordement_flottant")));
        }
    }


    @Override
    protected String getOperatorName() {
        return "*";
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {

        getLeftOperand().codeGenExprARM(compiler, register);
        compiler.getARMProgram().addInstruction(
            new ARMPush(register)
        );

        getRightOperand().codeGenExprARM(compiler, register);
        compiler.getARMProgram().addInstruction(
            new ARMPop(ARMRegister.R1)
        );

        compiler.getARMProgram().addInstruction(
            new ARMMul(register, ARMRegister.R1, register)
        );
    }

}
