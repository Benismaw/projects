package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;

import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMLabel;
import fr.ensimag.ARM.ARMLabelOperand;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMB;
import fr.ensimag.ARM.Instructions.ARMBne;
import fr.ensimag.ARM.Instructions.ARMCmp;
import fr.ensimag.ARM.Instructions.ARMLdr;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMStr;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.context.EnvironmentExp;
import fr.ensimag.deca.context.Type;
import fr.ensimag.deca.context.VariableDefinition;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.Register;

/**
 * @author gl16
 * @date 01/01/2026
 */
public class DeclVar extends AbstractDeclVar {

    
    final private AbstractIdentifier type;
    final private AbstractIdentifier varName;
    final private AbstractInitialization initialization;

    public DeclVar(AbstractIdentifier type, AbstractIdentifier varName, AbstractInitialization initialization) {
        Validate.notNull(type);
        Validate.notNull(varName);
        Validate.notNull(initialization);
        this.type = type;
        this.varName = varName;
        this.initialization = initialization;
    }

    @Override
    protected void verifyDeclVar(DecacCompiler compiler,
            EnvironmentExp localEnv, ClassDefinition currentClass)
            throws ContextualError {

        // On vérifie le type
        Type typeVar = type.verifyType(compiler);

        // Regle 3.17 pas de type void
        if (typeVar.isVoid()) {
            throw new ContextualError("Le type d'une variable ne peut pas être void (règle 3.17)", getLocation());
        }

        // On vérifie l'initialisation
        initialization.verifyInitialization(compiler, typeVar, localEnv, currentClass);

        // Regle 3.17 Variable deja declaré
        if (localEnv.get(varName.getName()) != null) {
            throw new ContextualError("Variable déjà déclarée dans ce bloc (règle 3.17)", getLocation());
        }

        // Ajout de la variable dans l'environnement local
        try {
            VariableDefinition varDef = new VariableDefinition(typeVar, getLocation());
            localEnv.declare(varName.getName(), varDef);

            // On lie l'identifiant à sa definition
            varName.setDefinition(varDef);


        } catch (EnvironmentExp.DoubleDefException e) {
            // Ne devrait jamais arriver
            throw new ContextualError("Variable déjà déclarée (règle 3.17)", getLocation());
        }

    }
    @Override
    protected void codeGenDecl(DecacCompiler compiler){
        // On demande une adresse libre dans la pile
        DAddr addr = compiler.getStackManager().allocGlobalVar();
        // On lie l'adresse à la definition de la variabler
        this.varName.getVariableDefinition().setOperand(addr);
        // Generation du code
        this.initialization.codeGenInitialization(compiler,Register.getR(2), addr);
    }
    
    @Override
    public void decompile(IndentPrintStream s) {
        type.decompile(s);
        s.print(" ");
        varName.decompile(s);
        initialization.decompile(s);
        s.print(";");
    }

    @Override
    protected
    void iterChildren(TreeFunction f) {
        type.iter(f);
        varName.iter(f);
        initialization.iter(f);
    }
    
    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        type.prettyPrint(s, prefix, false);
        varName.prettyPrint(s, prefix, false);
        initialization.prettyPrint(s, prefix, true);
    }

    @Override
protected void codeGenDeclARM(DecacCompiler compiler) {

    var prog = compiler.getARMProgram();
    String name = varName.getName().getName();
    Type varType = type.getDefinition().getType();

    
    if (varType.isFloat()) {
        prog.addData(name + ": .float 0.0");
    } else if (varType.isInt() && initialization instanceof Initialization) {
        AbstractExpr expr = ((Initialization) initialization).getExpression();
        if (expr instanceof IntLiteral) {
            prog.addData(name + ": .word " + ((IntLiteral) expr).getValue());
            return;
        } else {
            prog.addData(name + ": .word 0");
        }
    } else {
        prog.addData(name + ": .word 0");
    }

    
    if (!(initialization instanceof Initialization)) return;

    AbstractExpr expr = ((Initialization) initialization).getExpression();

    
    if (varType.isFloat()) {

        expr.codeGenExprARM(compiler, ARMRegister.R0);

        if (expr.getType().isInt()) {
            prog.addRaw("""
                vmov s0, r0
                vcvt.f32.s32 s0, s0
            """);
        } else {
            prog.addRaw("""
                vmov s0, r0
            """);
        }

        prog.addInstruction(
            new ARMLdr(
                ARMRegister.R1,
                new ARMLabelOperand(new ARMLabel("=" + name))
            )
        );
        prog.addRaw("    vstr.f32 s0, [r1]");
        return;
    }

    
    if (varType.isBoolean()) {
        expr.codeGenExprARM(compiler, ARMRegister.R0);

        String lTrue = "init_true_" + hashCode();
        String lEnd  = "init_end_" + hashCode();

        prog.addInstruction(new ARMCmp(ARMRegister.R0, new ARMImmediate(0)));
        prog.addInstruction(new ARMBne(lTrue));
        prog.addInstruction(new ARMMov(ARMRegister.R0, new ARMImmediate(0)));
        prog.addInstruction(new ARMB(lEnd));
        prog.addInstruction(new ARMLabel(lTrue));
        prog.addInstruction(new ARMMov(ARMRegister.R0, new ARMImmediate(1)));
        prog.addInstruction(new ARMLabel(lEnd));

        prog.addInstruction(
            new ARMLdr(
                ARMRegister.R1,
                new ARMLabelOperand(new ARMLabel("=" + name))
            )
        );
        prog.addInstruction(new ARMStr(ARMRegister.R0, ARMRegister.R1));
        return;
    }

    
    expr.codeGenExprARM(compiler, ARMRegister.R0);
    prog.addInstruction(
        new ARMLdr(
            ARMRegister.R1,
            new ARMLabelOperand(new ARMLabel("=" + name))
        )
    );
    prog.addInstruction(new ARMStr(ARMRegister.R0, ARMRegister.R1));
}

}


