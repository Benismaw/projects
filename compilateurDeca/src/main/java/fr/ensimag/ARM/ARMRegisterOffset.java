package fr.ensimag.ARM;

/**
 * Operand representing a register indirection with offset, e.g. 42(R3).
 *
 * @author Ensimag
 * @date 01/01/2026
 */
public class ARMRegisterOffset extends ARMDAddr {
    public int getOffset() {
        return offset;
    }
    public ARMRegister getRegister() {
        return register;
    }
    private final int offset;
    private final ARMRegister register;
    public ARMRegisterOffset(int offset, ARMRegister register) {
        super();
        this.offset = offset;
        this.register = register;
    }
    @Override
    public String toString() {
        return offset + "(" + register + ")";
    }
}
