package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMDAddr;
import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.ARMRegisterOffset;
import fr.ensimag.ARM.Instructions.ARMLdr;
import fr.ensimag.ARM.Instructions.ARMStr;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.STORE;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class Initialization extends AbstractInitialization {

    public AbstractExpr getExpression() {
        return expression;
    }

    private AbstractExpr expression;

    public void setExpression(AbstractExpr expression) {
        Validate.notNull(expression);
        this.expression = expression;
    }

    public Initialization(AbstractExpr expression) {
        Validate.notNull(expression);
        this.expression = expression;
    }

    @Override
    protected void verifyInitialization(DecacCompiler compiler, Type t,
            EnvironmentExp localEnv, ClassDefinition currentClass)
            throws ContextualError {

        // Regle 3.28 compatibilite
        AbstractExpr expression = getExpression().verifyRValue(compiler, localEnv, currentClass, t);

        this.setExpression(expression);
    }


    @Override
    public void decompile(IndentPrintStream s) {
        s.print(" = ");
        expression.decompile(s);
    }

    @Override
    protected
    void iterChildren(TreeFunction f) {
        expression.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        expression.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenInitialization(DecacCompiler compiler, GPRegister register, DAddr addr){
        expression.codeGenExpr(compiler, register);//on genere le code pour calculer l'expression et le resultat est placé dans register
        
        compiler.addInstruction(new STORE(register,addr));
    }

    @Override
    public void codeGenField(DecacCompiler compiler, int index){
        expression.codeGenExpr(compiler, Register.R0);//on calcule la valuer de l'expression dans R0

        compiler.addInstruction(new LOAD(new RegisterOffset(-2,Register.LB), Register.R1));//on recupere l'adresse de l'objet courant
        compiler.addInstruction(new STORE(Register.R0, new RegisterOffset(index, Register.R1)));// on stocke la valeur calculée dans le tas à l'offset index
    }

    @Override
    protected void codeGenInitializationARM(DecacCompiler compiler, ARMGPRegister register, ARMDAddr addr) {
        expression.codeGenExprARM(compiler, register);//on genere le code pour calculer l'expression et le resultat est placée dans register
        
        compiler.getARMProgram().addInstruction(new ARMStr(register, addr));
    }

    @Override
    protected void codeGenFieldARM(DecacCompiler compiler, int index) {
    expression.codeGenExprARM(compiler, ARMGPRegister.R0);
 
    // On utilise le registre R11 comme pointeur vers l'objet courant (frame pointer)
    compiler.getARMProgram().addInstruction(new ARMLdr(ARMGPRegister.R1, new ARMRegisterOffset(-2, ARMRegister.R11))); // Charger l'adresse de l'objet courant

    // Multiplication par 4 pour passer d'un index de mot à un offset en octets
    compiler.getARMProgram().addInstruction(new ARMStr(ARMGPRegister.R0, new ARMRegisterOffset(index * 4, ARMRegister.R1)));
    }


    
}
