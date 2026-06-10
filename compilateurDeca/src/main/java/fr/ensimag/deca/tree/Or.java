package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMBne;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.instructions.BNE;
import fr.ensimag.ima.pseudocode.instructions.CMP;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Or extends AbstractOpBool {

    public Or(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected String getOperatorName() {
        return "||";
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        // Calcul de l'operande gauche
        getLeftOperand().codeGenExpr(compiler, register);

        // On verifie si c'est vrai
        compiler.addInstruction(new CMP(new ImmediateInteger(0), register));

        Label finOr =  compiler.getLabelManager().uniqueLabel("fin_or");

        compiler.addInstruction(new BNE(finOr));

        // Operande Gauche est fausse
        getRightOperand().codeGenExpr(compiler, register);

        compiler.addLabel(finOr);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        throw new UnsupportedOperationException("Ne jamais appeler writeOperation sur un Or");
    }

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        var prog = compiler.getARMProgram();
        
        String labelEnd = "fin_or_" + this.hashCode();

        getLeftOperand().codeGenExprARM(compiler, register);
        prog.addInstruction(new ARMMov(ARMRegister.R1, new ARMImmediate(0)));

        prog.addInstruction(new ARMCmp(ARMRegister.R0, ARMRegister.R1));

        prog.addInstruction(new ARMBne(labelEnd));

        getRightOperand().codeGenExprARM(compiler, register);

        prog.addInstruction(new ARMLabel(labelEnd));
    }

}
