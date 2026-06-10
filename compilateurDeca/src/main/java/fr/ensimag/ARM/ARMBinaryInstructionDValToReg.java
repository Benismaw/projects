package fr.ensimag.ARM;

/**
 * Base class for instructions with 2 operands, the first being a
 * DVal, and the second a Register.
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMBinaryInstructionDValToReg extends ARMBinaryInstruction {
    public ARMBinaryInstructionDValToReg(ARMGPRegister op1, ARMDVal op2) {
        super(op1, op2);
    }
}
