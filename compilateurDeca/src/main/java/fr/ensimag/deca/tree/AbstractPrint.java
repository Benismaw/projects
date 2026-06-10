package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMBl;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;

/**
 * Print statement (print, println, ...).
 *
 * @author gl16
 * @date 01/01/2026
 */
public abstract class AbstractPrint extends AbstractInst {

    private boolean printHex;
    private ListExpr arguments = new ListExpr();
    
    abstract String getSuffix();

    public AbstractPrint(boolean printHex, ListExpr arguments) {
        Validate.notNull(arguments);
        this.arguments = arguments;
        this.printHex = printHex;
    }

    public ListExpr getArguments() {
        return arguments;
    }

    @Override
    protected void verifyInst(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass, Type returnType)
            throws ContextualError {
        for (AbstractExpr e: arguments.getList()) {
            Type t = e.verifyExpr(compiler, localEnv, currentClass);

            if (!(t.isString() || t.isInt() || t.isFloat()))  {
                throw new ContextualError("Type" + t + " non supporté par print",e.getLocation());
            }
        }
    }

    @Override
    protected void codeGenInst(DecacCompiler compiler) {
        for (AbstractExpr a : getArguments().getList()) {
            if (getPrintHex()) {
                a.codeGenPrintHex(compiler);
            }
            else  {
                a.codeGenPrint(compiler);
            }
        }
    }

    private boolean getPrintHex() {
        return printHex;
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("print");
        s.print(getSuffix());
        if(getPrintHex())
            s.print("x");
        s.print("(");
        boolean first = true;
        for (AbstractExpr e : arguments.getList()) {
            if (!first) {
                s.print(", ");
            }
            first = false;
            e.decompile(s);
        }
        s.print(");");
    }

    @Override
    protected void iterChildren(TreeFunction f) {
        arguments.iter(f);
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        arguments.prettyPrint(s, prefix, true);
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        for (AbstractExpr expr : getArguments().getList()) {
        expr.codeGenExprARM(compiler, ARMRegister.R0);
        compiler.getARMProgram().addInstruction(
            new ARMBl("__print_int")
        );
    }
    }

}
