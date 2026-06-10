package fr.ensimag.deca.tree;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.tools.IndentPrintStream;
import java.io.PrintStream;

import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.POP;
import fr.ensimag.ima.pseudocode.instructions.PUSH;
import org.apache.commons.lang.Validate;

/**
 * Binary expressions.
 *
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractBinaryExpr extends AbstractExpr {

    public AbstractExpr getLeftOperand() {
        return leftOperand;
    }

    public AbstractExpr getRightOperand() {
        return rightOperand;
    }

    protected void setLeftOperand(AbstractExpr leftOperand) {
        Validate.notNull(leftOperand);
        this.leftOperand = leftOperand;
    }

    protected void setRightOperand(AbstractExpr rightOperand) {
        Validate.notNull(rightOperand);
        this.rightOperand = rightOperand;
    }

    private AbstractExpr leftOperand;
    private AbstractExpr rightOperand;

    public AbstractBinaryExpr(AbstractExpr leftOperand,
            AbstractExpr rightOperand) {
        Validate.notNull(leftOperand, "left operand cannot be null");
        Validate.notNull(rightOperand, "right operand cannot be null");
        Validate.isTrue(leftOperand != rightOperand, "Sharing subtrees is forbidden");
        this.leftOperand = leftOperand;
        this.rightOperand = rightOperand;
    }


    @Override
    public void decompile(IndentPrintStream s) {
        s.print("(");
        getLeftOperand().decompile(s);
        s.print(" " + getOperatorName() + " ");
        getRightOperand().decompile(s);
        s.print(")");
    }

    protected abstract void writeOperation(DecacCompiler compiler, GPRegister register, GPRegister nextRegister);

    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {
        // On mets la partie gauche dans le meme registre que le resultat final
        getLeftOperand().codeGenExpr(compiler, register);

        // On recupère le numéro du registre actuel et la limite de registre imposé
        int n = register.getNumber();
        int maxReg = compiler.getCompilerOptions().getRegisters();

        if (n < maxReg-1) {
            GPRegister nextRegister = GPRegister.getR(n+1);
            // On evalue la partie droite
            getRightOperand().codeGenExpr(compiler, nextRegister);
            // On realise l'operation
            writeOperation(compiler, register, nextRegister);

            // Plus de registre disponible, on sauvegarde dans la pile
        } else {
            // Sauvegarde R15 sur la pile
            compiler.addInstruction(new PUSH(register));
            compiler.getStackManager().increment(1);

            //On evalue la partie droite dans R15
            getRightOperand().codeGenExpr(compiler, register);

            // On deplace le resultat de la partie droite dans R0
            compiler.addInstruction(new LOAD(register, Register.R0));

            // On restore la partie gauche qui est sur la pile dans R15
            compiler.addInstruction(new POP(register));
            compiler.getStackManager().decrement(1);

            // On realise l'operation
            writeOperation(compiler, register, Register.R0);
        }
    }

    abstract protected String getOperatorName();

    @Override
    protected void iterChildren(TreeFunction f) {
        leftOperand.iter(f);
        rightOperand.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        leftOperand.prettyPrint(s, prefix, false);
        rightOperand.prettyPrint(s, prefix, true);
    }

}
