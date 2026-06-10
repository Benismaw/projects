package fr.ensimag.deca.tree;

import java.io.PrintStream;

import org.apache.commons.lang.Validate;
import org.apache.log4j.Logger;

import fr.ensimag.ARM.ARMImmediate;
import fr.ensimag.ARM.ARMRegister;
import fr.ensimag.ARM.Instructions.ARMMov;
import fr.ensimag.ARM.Instructions.ARMSvc;
import fr.ensimag.deca.DecacCompiler;
import fr.ensimag.deca.codegen.arm.ARMProgram;
import fr.ensimag.deca.context.ClassDefinition;
import fr.ensimag.deca.context.ContextualError;
import fr.ensimag.deca.tools.IndentPrintStream;
import fr.ensimag.ima.pseudocode.DAddr;
import fr.ensimag.ima.pseudocode.Label;
import fr.ensimag.ima.pseudocode.LabelOperand;
import fr.ensimag.ima.pseudocode.NullOperand;
import fr.ensimag.ima.pseudocode.Register;
import fr.ensimag.ima.pseudocode.RegisterOffset;
import fr.ensimag.ima.pseudocode.instructions.ADDSP;
import fr.ensimag.ima.pseudocode.instructions.BOV;
import fr.ensimag.ima.pseudocode.instructions.ERROR;
import fr.ensimag.ima.pseudocode.instructions.HALT;
import fr.ensimag.ima.pseudocode.instructions.LOAD;
import fr.ensimag.ima.pseudocode.instructions.STORE;
import fr.ensimag.ima.pseudocode.instructions.TSTO;
import fr.ensimag.ima.pseudocode.instructions.WINT;
import fr.ensimag.ima.pseudocode.instructions.WNL;
import fr.ensimag.ima.pseudocode.instructions.WSTR;

/**
 * Deca complete program (class definition plus main block)
 *
 * @author gl16
 * @date 01/01/2026
 */
public class Program extends AbstractProgram {
    private static final Logger LOG = Logger.getLogger(Program.class);
    
    public Program(ListDeclClass classes, AbstractMain main) {
        Validate.notNull(classes);
        Validate.notNull(main);
        this.classes = classes;
        this.main = main;
    }
    public ListDeclClass getClasses() {
        return classes;
    }
    public AbstractMain getMain() {
        return main;
    }
    private ListDeclClass classes;
    private AbstractMain main;

    @Override
    public void verifyProgram(DecacCompiler compiler) throws ContextualError {
        LOG.debug("verify program: start");

        // Passe 1
        classes.verifyListClass(compiler);
        // Passe 2
        classes.verifyListClassMembers(compiler);
        // Passe 3
        classes.verifyListClassBody(compiler);

        main.verifyMain(compiler);
        LOG.debug("verify program: end");
    }

@Override
public void codeGenProgram(DecacCompiler compiler) {

        if (compiler.getCompilerOptions().isARM()) {
            codeGenProgramARM(compiler);
            return;
        }
    compiler.getStackManager().reset();
    compiler.addComment("Construction des tables des méthodes");

    // Générer manuellement la VTable d'Object (fondation de l'héritage)
    ClassDefinition objectDef = (ClassDefinition) compiler.environmentType.defOfType(compiler.createSymbol("Object"));
    DAddr objectVtableAddr = compiler.getStackManager().allocGlobalVar(); // Offset 1(GB) pour le lien super
    compiler.getStackManager().allocGlobalVar(); // Offset 2(GB) pour equals
    objectDef.setVtableAddr(objectVtableAddr);

    compiler.addInstruction(new LOAD(new NullOperand(), Register.R0));
    compiler.addInstruction(new STORE(Register.R0, objectVtableAddr)); // Pointeur super = null

    compiler.addInstruction(new LOAD(new LabelOperand(new Label("code.Object.equals")), Register.R0));
    compiler.addInstruction(new STORE(Register.R0, new RegisterOffset(1, Register.LB))); // LB pointe sur GB au début

    // Générer les VTables des classes utilisateur
    classes.codeGenListDeclClassVTable(compiler);

    // Reste du programme
    main.codeGenMain(compiler);
    compiler.addInstruction(new HALT());

    // Blocs de gestion d'erreurs 
    if (!compiler.getCompilerOptions().getNoCheck()) {
        addErrorBlock(compiler, "division_zero", "division par zéro");
        addErrorBlock(compiler, "debordement_flottant", "Débordement flottant");
        addErrorBlock(compiler, "pile_pleine", "Débordement de la pile");
        addErrorBlock(compiler, "tas_plein", "Tas plein");
        addErrorBlock(compiler, "dereferencement_null", "Déréférencement null");
        addErrorBlock(compiler, "cast_error", "Cast impossible");
        addErrorBlock(compiler, "io_error", "Erreur io");
    }

    // Génération des méthodes et des initialiseurs (Hors du flux principal)
    classes.codeGenListDeclClassBody(compiler); 

    // Calcul final des ressources pour le préambule 
    // getNbGlobalVars() inclut TOUS les slots GB (VTables + variables du Main)
    int totalGlobalSlots = compiler.getStackManager().getNbGlobalVars(); 
    int maxTemp = compiler.getStackManager().getMaxStack();
    int d = totalGlobalSlots + maxTemp;

    //Insertion du préambule (en ordre inverse car addFirstInstruction empile en haut)
    if (totalGlobalSlots > 0) {
        compiler.addFirstInstruction(new ADDSP(totalGlobalSlots));
    }
    
    if (!compiler.getCompilerOptions().getNoCheck()) {
        compiler.addFirstInstruction(new BOV(new Label("pile_pleine")));
        compiler.addFirstInstruction(new TSTO(d));
    } else {
        compiler.addFirstInstruction(new TSTO(d));
    }
}
    @Override
    public void decompile(IndentPrintStream s) {
        getClasses().decompile(s);
        getMain().decompile(s);
    }

    /**
     * Génère un bloc d'erreur standardisé
     */
    private void addErrorBlock(DecacCompiler compiler, String labelName, String errorMessage) {
        compiler.addLabel(new Label(labelName));
        compiler.addInstruction(new WSTR("Erreur ligne "));
        loadErrorLine(compiler);
        compiler.addInstruction(new WINT());
        compiler.addInstruction(new WSTR(": " + errorMessage));
        compiler.addInstruction(new WNL());
        compiler.addInstruction(new ERROR());
    }
    
    @Override
    protected void iterChildren(TreeFunction f) {
        classes.iter(f);
        main.iter(f);
    }
    @Override
    protected void prettyPrintChildren(PrintStream s, String prefix) {
        classes.prettyPrint(s, prefix, false);
        main.prettyPrint(s, prefix, true);
    }

    @Override
    public void codeGenProgramARM(DecacCompiler compiler) {

        ARMProgram program = compiler.getARMProgram();

        // generation du main 
        main.codeGenMainARM(compiler);

        // fin du programme : exit(0)
        program.addInstruction((new ARMMov(ARMRegister.R7, new ARMImmediate(1))));
        program.addInstruction(new ARMMov(ARMRegister.R0, new ARMImmediate(0)));
        program.addInstruction(new ARMSvc(0));

                        
    }   
}
