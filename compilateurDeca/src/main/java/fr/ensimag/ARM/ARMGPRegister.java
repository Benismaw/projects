package fr.ensimag.ARM;

/**
 * General Purpose Register operand (R0, R1, ... R15).
 * 
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMGPRegister extends ARMRegister {
    public static final String LB = null;

    /**
     * @return the number of the register, e.g. 12 for R12.
     */
    public int getNumber() {
        return number;
    }

    private int number;

    ARMGPRegister(String name, int number) {
        super(name);
        this.number = number;
    }
}
