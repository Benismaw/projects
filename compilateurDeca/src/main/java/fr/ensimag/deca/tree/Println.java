package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMPrintFloat;
import fr.ensimag.ARM.Instructions.ARMPrintInt;
import fr.ensimag.ARM.Instructions.ARMPrintString;
import fr.ensimag.ARM.Instructions.ARMPrintln;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.ima.pseudocode.instructions.WNL;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class Println extends AbstractPrint {

    /**
     * @param arguments arguments passed to the print(...) statement.
     * @param printHex if true, then float should be displayed as hexadecimal (printlnx)
     */
    public Println(boolean printHex, ListExpr arguments) {
        super(printHex, arguments);
    }

    @Override
    protected void codeGenInst(DecacCompiler compiler) {
        super.codeGenInst(compiler);
        compiler.addInstruction(new WNL());
    }

    @Override
    String getSuffix() {
        return "ln";
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        // génération du code pour chaque argument
        for(AbstractExpr arg : getArguments().getList()) {

            arg.codeGenExprARM(compiler, ARMGPRegister.R0);

            if (arg.getType().isInt()) {
                compiler.getARMProgram().addInstruction(
                    new ARMPrintInt()
                );
            } else if (arg.getType().isFloat()) {
                compiler.getARMProgram().addInstruction(
                    new ARMPrintFloat()
                );
            } else if (arg.getType().isString()) {
                arg.codeGenExprARM(compiler, ARMRegister.R0); 
                compiler.getARMProgram().addInstruction(new ARMPrintString());
            } else {
                // inutile si la verification contextuelle est correcte
                throw new UnsupportedOperationException(
                    "print ARM non supporté par le type " + arg.getType()
                );
            }
        }
        // appel de l instruction ARMPrinln
        compiler.getARMProgram().addInstruction(new ARMPrintln());
    }
}
