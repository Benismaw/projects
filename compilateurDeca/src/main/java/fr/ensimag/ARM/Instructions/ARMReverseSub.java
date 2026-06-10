package fr.ensimag.ARM.Instructions;
import fr.ensimag.ARM.*;

public class ARMReverseSub extends ARMInstruction {
    
    private ARMRegister dst;
    private ARMRegister src;

    public ARMReverseSub(ARMRegister dst, ARMRegister src) {
        this.dst = dst;
        this.src = src;
    }

    @Override
    public String toString() {
        return "rsb " + dst + ", " + src + ", #0";
    }
}
