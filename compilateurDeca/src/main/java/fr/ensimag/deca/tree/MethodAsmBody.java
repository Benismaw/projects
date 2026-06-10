package fr.ensimag.deca.tree;

import java.io.PrintStream;

import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.InlinePortion;

public class MethodAsmBody extends AbstractMethodBody {
    private StringLiteral codeAsm;
    private ListInst insts;

    public MethodAsmBody(StringLiteral codeAsm) {
        this.codeAsm=codeAsm;
    }

    protected void verifyMethodBody(DecacCompiler compiler, 
            EnvironmentExp localEnv, ClassDefinition currentClass, Type returnType) 
            throws ContextualError {
        //en passe 3 on verifie simplement le litteral de chaine
        codeAsm.verifyExpr(compiler, localEnv, currentClass);
    }

    @Override
    public void codeGenMethodBody(DecacCompiler compiler) {
        String asm = codeAsm.getValue();

        if (asm.startsWith("\"") && asm.endsWith("\"")) {
            asm = asm.substring(1, asm.length() - 1);
        }
        asm = asm.replace("\\\"", "\"").replace("\\n", "\n");

        compiler.add(new InlinePortion(asm));
    }

    public void decompile(IndentPrintStream s) {
        s.print("asm(");
        codeAsm.decompile(s);
        s.print(");");
    }

    protected void iterChildren(TreeFunction f) {
        codeAsm.iter(f);
    }

    protected void prettyPrintChildren(PrintStream s, String prefix) {
        codeAsm.prettyPrint(s, prefix, true);
    }
}