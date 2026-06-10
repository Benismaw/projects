package fr.ensimag.deca.tree;


import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMBeq;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMDiv;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMPop;
import fr.ensimag.ARM.Instructions.ARMPush;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.ImmediateFloat;
import fr.ensimag.ima.pseudocode.ImmediateInteger;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.BEQ;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.CMP;
import fr.ensimag.ima.pseudocode.instructions.DIV;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;
import fr.ensimag.ima.pseudocode.instructions.QUO;

/**
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Divide extends AbstractOpArith {
    public Divide(AbstractExpr leftOperand, AbstractExpr rightOperand) {
        super(leftOperand, rightOperand);
    }

    @Override
    protected void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister) {

        boolean noCheck = compiler.getCompilerOptions().getNoCheck();

        if (getType().isInt()) {
            if (!noCheck) {
                compiler.addInstruction(new CMP(new ImmediateInteger(0), nextRegister));
                compiler.addInstruction(new BEQ(new Label("division_zero")));
            }
            compiler.addInstruction(new QUO(nextRegister, register));

        // Division flottante
        } else if (getType().isFloat()) {
            if (!noCheck) {
                compiler.addInstruction(new CMP(new ImmediateFloat(0f), nextRegister));
                compiler.addInstruction(new BEQ(new Label("division_zero")));
            }
            compiler.addInstruction(new DIV(nextRegister, register));
            if (!noCheck) {
                compiler.addInstruction(new BOV(new Label("debordement_flottant")));
            }
        }

    }

@Override
protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
    // Évaluer la GAUCHE d'abord 
    getLeftOperand().codeGenExpr(compiler, register);

    // Sauvegarder le résultat de gauche sur la pile pour le protége
    compiler.addInstruction(new PUSH(register));

    // Évaluer la DROITE 

    getRightOperand().codeGenExpr(compiler, Register.R0);    // On utilise R0 comme registre temporaire pour la droite

    // Récupérer la gauche dans R1 
    compiler.addInstruction(new POP(Register.R1));

    // Effectuer l'opération : GAUCHE (R1) / DROITE (R0)
    if (getType().isFloat()) {
        compiler.addInstruction(new DIV(Register.R0, Register.R1));
        if (!compiler.getCompilerOptions().getNoCheck()) {
            compiler.addInstruction(new BOV(new Label("debordement_flottant")));
        }
    } else {
        compiler.addInstruction(new QUO(Register.R0, Register.R1));
        if (!compiler.getCompilerOptions().getNoCheck()) {
            compiler.addInstruction(new BOV(new Label("division_zero")));
        }
    }

    // Charger le résultat final dans le registre de destination
    compiler.addInstruction(new LOAD(Register.R1, register));
}


    @Override
    protected String getOperatorName() {
        return "/";
    }

    private boolean isStaticallyZero(AbstractExpr expr) {
        if (expr instanceof IntLiteral) {
            return ((IntLiteral) expr).getValue() == 0;
        }

        if (expr instanceof Plus) {
            Plus p = (Plus) expr;
            return isStaticallyZero(p.getLeftOperand()) && isStaticallyZero(p.getRightOperand());
        }

        if (expr instanceof Minus) {
            Minus m = (Minus) expr;
            return isStaticallyZero(m.getLeftOperand()) && isStaticallyZero(m.getRightOperand());
        }

        if (expr instanceof Multiply) {
            Multiply m = (Multiply) expr;
            return isStaticallyZero(m.getLeftOperand()) || isStaticallyZero(m.getRightOperand());
        }

        return false;
    }


    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {

    var prog = compiler.getARMProgram();
    
    getLeftOperand().codeGenExprARM(compiler, register);
    prog.addInstruction(new ARMPush(register));

    
    getRightOperand().codeGenExprARM(compiler, register);
    prog.addInstruction(new ARMMov(ARMRegister.R1, register));
    
    //recuperer le gauche
    prog.addInstruction(new ARMPop(ARMRegister.R0));

    if (!compiler.getCompilerOptions().getNoCheck()) {
        prog.addInstruction(new ARMCmp(ARMRegister.R1, new ARMImmediate(0)));
        prog.addInstruction(new ARMBeq("division_zero"));
    }

    prog.addInstruction(new ARMDiv());

    if (register != ARMRegister.R0) {
            prog.addInstruction(new ARMMov(register, ARMRegister.R0));
        }
}

}
