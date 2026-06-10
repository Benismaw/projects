package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.ARM.Instructions.ARMSub;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;
import fr.ensimag.ima.pseudocode.instructions.SUB;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class Minus extends AbstractOpArith {
    public Minus(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {
        compiler.addInstruction(new SUB(nextRegister, register));

        if (getType().isFloat() && !compiler.getCompilerOptions().getNoCheck()) {
            compiler.addInstruction(new BOV(new Label("debordement_flottant")));
        }
    }

@Override
protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
    // Évaluer d'abord le membre de GAUCHE 
    getLeftOperand().codeGenExpr(compiler, register);

    // Sauvegarder le résultat de gauche sur la pile
    compiler.addInstruction(new PUSH(register));// Cela le protège contre tout écrasement par un appel de méthode récursif à droite


    // evaluer le membre de DROITE
    getRightOperand().codeGenExpr(compiler, Register.R0);

    // récupérer la valeur de gauche dans R1
    compiler.addInstruction(new POP(Register.R1));

    // effectuer la soustraction : R1 = R1 - R0 
    compiler.addInstruction(new SUB(Register.R0, Register.R1));

    // placer le résultat final dans le registre de destination
    compiler.addInstruction(new LOAD(Register.R1, register));

    // Gestion du débordement pour les flottants
    if (getType().isFloat() && !compiler.getCompilerOptions().getNoCheck()) {
        compiler.addInstruction(new BOV(new Label("debordement_flottant")));
    }
}



    @Override
    protected String getOperatorName() {
        return "-";
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

        // R0 = R1 - R0
        compiler.getARMProgram().addInstruction(
            new ARMSub(register, ARMRegister.R1, register)
        );
    }
    
}
