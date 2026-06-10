package fr.ensimag.deca.tree;

import fr.ensimag.ARM.ARMGPRegister;
import fr.ensimag.ARM.Instructions.ARMPrintFloat;
import fr.ensimag.ARM.Instructions.ARMPrintInt;
import fr.ensimag.ARM.Instructions.ARMPrintString;
import fr.ensimag.deca.DecacCompiler;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class Print extends AbstractPrint {
    /**
     * @param arguments arguments passed to the print(...) statement.
     * @param printHex if true, then float should be displayed as hexadecimal (printx)
     */
    public Print(boolean printHex, ListExpr arguments) {
        super(printHex, arguments);
    }

    @Override
    String getSuffix() {
        return "";
    }

    @Override
    protected void codeGenInstARM(DecacCompiler compiler) {
        
        for(AbstractExpr arg : getArguments().getList()) {

            arg.codeGenExprARM(compiler, ARMGPRegister.R1);

            if (arg.getType().isInt()) {
                compiler.getARMProgram().addInstruction(
                    new ARMPrintInt()
                );
            } else if (arg.getType().isFloat()) {
                compiler.getARMProgram().addInstruction(
                    new ARMPrintFloat()
                );
            } else if (arg.getType().isString()) {
                compiler.getARMProgram().addInstruction(
                    new ARMPrintString()
                );
            } else {
                // inutile si la verification contextuelle est correcte
                throw new UnsupportedOperationException(
                    "print ARM non supporté par le type " + arg.getType()
                );
            }
        }

        
        }

    }

    

