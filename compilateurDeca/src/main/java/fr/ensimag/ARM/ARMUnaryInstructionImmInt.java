package fr.ensimag.ARM;

/**
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public abstract class ARMUnaryInstructionImmInt extends ARMUnaryInstruction {

    protected ARMUnaryInstructionImmInt(ARMImmediateInteger operand) {
        super(operand);
    }

    protected ARMUnaryInstructionImmInt(int i) {
        super(new ARMImmediateInteger(i));
    }

}
