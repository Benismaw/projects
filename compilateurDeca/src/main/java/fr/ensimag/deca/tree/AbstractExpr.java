package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.DecacInternalError;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.WFLOAT;
import fr.ensimag.ima.pseudocode.instructions.WFLOATX;
import fr.ensimag.ima.pseudocode.instructions.WINT;

/**
 * Expression, i.e. anything that has a value.
 *
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractExpr extends AbstractInst {
    /**
     * @return true if the expression does not correspond to any concrete token
     * in the source code (and should be decompiled to the empty string).
     */
    boolean isImplicit() {
        return false;
    }

    /**
     * Get the type decoration associated to this expression (i.e. the type computed by contextual verification).
     */
    public Type getType() {
        return type;
    }

    protected void setType(Type type) {
        Validate.notNull(type);
        this.type = type;
    }
    private Type type;

    @Override
    protected void checkDecoration() {
        if (getType() == null) {
            throw new DecacInternalError("Expression " + decompile() + " has no Type decoration");
        }
    }

    /**
     * Verify the expression for contextual error.
     * 
     * implements non-terminals "expr" and "lvalue" 
     *    of [SyntaxeContextuelle] in pass 3
     *
     * @param compiler  (contains the "env_types" attribute)
     * @param localEnv
     *            Environment in which the expression should be checked
     *            (corresponds to the "env_exp" attribute)
     * @param currentClass
     *            Definition of the class containing the expression
     *            (corresponds to the "class" attribute)
     *             is null in the main bloc.
     * @return the Type of the expression
     *            (corresponds to the "type" attribute)
     */
    public abstract Type verifyExpr(DecacCompiler compiler,
            EnvironmentExp localEnv, ClassDefinition currentClass)
            throws ContextualError;

    /**
     * Verify the expression in right hand-side of (implicit) assignments 
     * 
     * implements non-terminal "rvalue" of [SyntaxeContextuelle] in pass 3
     *
     * @param compiler  contains the "env_types" attribute
     * @param localEnv corresponds to the "env_exp" attribute
     * @param currentClass corresponds to the "class" attribute
     * @param expectedType corresponds to the "type1" attribute            
     * @return this with an additional ConvFloat if needed...
     */
    public AbstractExpr verifyRValue(DecacCompiler compiler,
            EnvironmentExp localEnv, ClassDefinition currentClass, 
            Type expectedType)
            throws ContextualError {

        Type typeRight = this.verifyExpr(compiler, localEnv, currentClass);

        if (!Type.assignCompatible(compiler.environmentType, expectedType, typeRight)) {
            throw new ContextualError("Type " + typeRight + " incompatible avec le type attendu " + expectedType + " (règle 3.28)", getLocation());
        }

        if (expectedType.isFloat() && typeRight.isInt()) {
            ConvFloat conv = new ConvFloat(this);
            conv.verifyExpr(compiler, localEnv, currentClass);
            return conv;
        }

        // Sous typage a faire objet


        return this;
    }
    
    
    @Override
    protected void verifyInst(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass, Type returnType)
            throws ContextualError {
        this.verifyExpr(compiler, localEnv, currentClass);
    }

    /**
     * Verify the expression as a condition, i.e. check that the type is
     * boolean.
     *
     * @param localEnv
     *            Environment in which the condition should be checked.
     * @param currentClass
     *            Definition of the class containing the expression, or null in
     *            the main program.
     */
    void verifyCondition(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        Type t = this.verifyExpr(compiler, localEnv, currentClass);
        
        if (this instanceof Assign) {
            throw new ContextualError("L'affectation est interdite dans une condition", this.getLocation());
    }
        // Regle 3.29
        if (!t.isBoolean()) {
            throw new ContextualError("Condition non booléenne (règle 3.29)", this.getLocation());
        }
    }

    /**
     * Generate code to print the expression
     *
     * @param compiler
     */
    protected void codeGenPrint(DecacCompiler compiler) {
        // Calculer la valeur de l'expression dans un registre temporaire
        codeGenExpr(compiler, Register.getR(2));


        // On deplace le resultat dans R1
        compiler.addInstruction(new LOAD(Register.getR(2), Register.R1));

        // On affiche selon le type
        if (getType().isInt()) {
            compiler.addInstruction(new WINT());
        } else if (getType().isFloat()) {
            compiler.addInstruction(new WFLOAT());
        }
    }

    /**
     * Génère le code pour l'affichage hexadécimal (printx).
     */
    protected void codeGenPrintHex(DecacCompiler compiler) {
        if (getType().isFloat()) {
            codeGenExpr(compiler, Register.getR(2));
            compiler.addInstruction(new LOAD(Register.getR(2), Register.R1));
            compiler.addInstruction(new WFLOATX());
        } else {
            codeGenPrint(compiler);
        }
    }

    @Override
    protected void codeGenInst(DecacCompiler compiler) {
        codeGenExpr(compiler, Register.getR(2));
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        codeGenExprARM(compiler, ARMRegister.getR(0));
    }

    protected abstract void codeGenExpr(DecacCompiler compiler, GPRegister register);
    
    protected abstract void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register);
    

    @Override
    protected void decompileInst(IndentPrintStream s) {
        decompile(s);
        s.print(";");
    }

    @Override
    protected void prettyPrintType(PrintStream s, String prefix) {
        Type t = getType();
        if (t != null) {
            s.print(prefix);
            s.print("type: ");
            s.print(t);
            s.println();
        }
    }
}
