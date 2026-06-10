package fr.ensimag.ARM;

/**
 * Base class for instructions with 2 operands, the first being a
 * DAddr, and the second a Register.
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMBinaryInstructionDAddrToReg extends ARMBinaryInstructionDValToReg {

    public ARMBinaryInstructionDAddrToReg(ARMDAddr op2, ARMGPRegister op1) {
        super(op1, op2);
    }

}
