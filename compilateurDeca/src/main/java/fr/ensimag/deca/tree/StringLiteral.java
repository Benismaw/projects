package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMLabelOperand;
import fr.ensimag.ARM.Instructions.ARMLdr;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.GPRegister;
import fr.ensimag.ima.pseudocode.instructions.WNL;
import fr.ensimag.ima.pseudocode.instructions.WSTR;

/**
 * String literal
 *
 * @author gl16
 * @date 01/01/2026
 */
public class StringLiteral extends AbstractStringLiteral {

    @Override
    public String getValue() {
        return value;
    }

    private String value;

    public StringLiteral(String value) {
        Validate.notNull(value);
        String s = value.substring(1, value.length() - 1);
        s = s.replace("\\\\", "\\");
        s = s.replace("\\\"", "\"");
        s = s.replace("\\t", "\t");
        this.value = s;
    }

    @Override
    public Type verifyExpr(DecacCompiler compiler, EnvironmentExp localEnv,
            ClassDefinition currentClass) throws ContextualError {
        this.setType(compiler.environmentType.STRING);
        return compiler.environmentType.STRING;
    }

    @Override
    protected void codeGenPrint(DecacCompiler compiler) {

        String s = this.value;
        String[] lines = s.split("\n", -1);


        for (int i = 0; i < lines.length; i++) {
            if (!lines[i].isEmpty()) {
                compiler.addInstruction(new WSTR(lines[i]));
            }
            if (i < lines.length - 1) {
                compiler.addInstruction(new WNL());
            }
        }
    }


    protected void codeGenPrintHex(DecacCompiler compiler) {
        codeGenPrint(compiler);
    }

    @Override
    protected void codeGenExpr(DecacCompiler compiler, GPRegister register) {

    }

    private String escape(String s) {
        return s
            .replace("\\", "\\\\")
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\t", "\\t");
    }

    @Override
    public void decompile(IndentPrintStream s) {
        s.print("\"");
        String escapedValue = value.replace("\\", "\\\\")
                .replace("\"", "\\\"");
        s.print(escapedValue);
        s.print("\"");
    }



    @Override
    protected void iterChildren(TreeFunction f) {
        // leaf node => nothing to do
    }

    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        // leaf node => nothing to do
    }
    
    @Override
    String prettyPrintNode() {
        return "StringLiteral (" + value + ")";
    }

    private static int stringCounter = 0;

    @Override
    protected void codeGenExprARM(DecacCompiler compiler, ARMGPRegister register) {
        String label = "str_" + (stringCounter++);
        compiler.getARMProgram().addData(
            label + ": .asciz \"" + escape(value) + "\""
        );
        compiler.getARMProgram().addInstruction(
            new ARMLdr(register, new ARMLabelOperand(new ARMLabel("=" + label)))
        );
    }

}
