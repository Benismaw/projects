package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMAdd;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.ADD;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class Plus extends AbstractOpArith {
    public Plus(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {

        compiler.addInstruction(new ADD(nextRegister, register));

        if (getType().isFloat() && !compiler.getCompilerOptions().getNoCheck()) {
            compiler.addInstruction(new BOV(new Label("debordement_flottant")));
        }
    }

@Override
protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
    // Évaluer d'abord le membre de GAUCHE 
    getLeftOperand().codeGenExpr(compiler, register);

    // Sauvegarder le résultat de gauche sur la pile pour le protéger
    compiler.addInstruction(new PUSH(register));

    // Évaluer le membre de DROITE
    
    getRightOperand().codeGenExpr(compiler, Register.R0);// On utilise R0 comme registre scratch pour stocker le résultat de droite

    // Récupérer la valeur de gauche dans R1 
    compiler.addInstruction(new POP(Register.R1));

    // Effectuer l'addition : R1 = R1 + R0
    compiler.addInstruction(new ADD(Register.R0, Register.R1));

    // Placer le résultat final dans le registre de destination
    compiler.addInstruction(new LOAD(Register.R1, register));

    // Gestion du débordement pour les flottants
    if (getType().isFloat() && !compiler.getCompilerOptions().getNoCheck()) {
        compiler.addInstruction(new BOV(new Label("debordement_flottant")));
    }
}
    @Override
    protected String getOperatorName() {
        return "+";
    }

    @Override 
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {

        getLeftOperand().codeGenExprARM(compiler, ARMGPRegister.R0);
        compiler.getARMProgram().addInstruction(
            new ARMPush(ARMRegister.R0)
        );

        getRightOperand().codeGenExprARM(compiler, ARMGPRegister.R0);
        compiler.getARMProgram().addInstruction(
            new ARMPop(ARMRegister.R1)
        );

        compiler.getARMProgram().addInstruction(
            new ARMAdd(register, ARMRegister.R1, ARMRegister.R0)
        );

    }
}
