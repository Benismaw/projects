package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.Instructions.ARMBeq;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.CMP;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class And extends AbstractOpBool {

    public And(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }


    @Override
    protected String getOperatorName() {
        return "&&";
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        // Calcul de l'operande gauche
        getLeftOperand().codeGenExpr(compiler, register);

        // On verifie si c'est faux
        compiler.addInstruction(new CMP(new ImmediateInteger(0), register));

        Label finAnd =  compiler.getLabelManager().uniqueLabel("fin_and");

        compiler.addInstruction(new BEQ(finAnd));

        // Operande Gauche est vrai
        getRightOperand().codeGenExpr(compiler, register);

        compiler.addLabel(finAnd);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        throw new UnsupportedOperationException("Ne jamais appeler writeOperation sur un And");
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();
        
        String labelEnd = "and_end_" + this.hashCode();
        //String labelFalse = "and_false_" + this.hashCode();

        getLeftOperand().codeGenExprARM(compiler, register);

        prog.addInstruction(new ARMCmp(register, new ARMImmediate(0)));

        prog.addInstruction(new ARMBeq(labelEnd));

        getRightOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMLabel(labelEnd));

    }
}
